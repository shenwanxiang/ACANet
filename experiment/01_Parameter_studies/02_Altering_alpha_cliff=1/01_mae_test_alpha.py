from math import sqrt
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy.stats import pearsonr
from tqdm import tqdm
from torch_geometric.loader import DataLoader


# import clsar package
import sys, os
sys.path.insert(0, '/home/shenwanxiang/Research/bidd-clsar/')
from clsar.model.loss import ACALoss, get_best_cliff
from clsar.model.model import ACANet_PNA, get_deg  # model
from clsar.feature import Gen39AtomFeatures  # feature
from clsar.dataset import LSSNS  # dataset


import os, random
import numpy as np
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


gpuid = 1
torch.cuda.set_device(gpuid)


def train(train_loader, device, optimizer, model, aca_loss):
    model.train()
    total_examples = 0
    total_loss = 0
    total_tsm_loss = 0
    total_reg_loss = 0
    n_triplets = []
    n_pos_triplets = []
    
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        predictions, embeddings = model(data.x.float(), data.edge_index,
                                        data.edge_attr, data.batch)

        loss_out = aca_loss(labels=data.y,
                            predictions=predictions,
                            embeddings=embeddings)

        loss, reg_loss, tsm_loss, n, n_pos = loss_out

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_tsm_loss += float(tsm_loss) * data.num_graphs
        total_reg_loss += float(reg_loss) * data.num_graphs
        total_examples += data.num_graphs

        n_triplets.append(int(n))
        n_pos_triplets.append(int(n_pos))

    train_loss = total_loss / total_examples
    total_tsm_loss = total_tsm_loss / total_examples
    total_reg_loss = total_reg_loss / total_examples
    n_triplets = int(sum(n_triplets) / (i+1))
    n_pos_triplets = int(sum(n_pos_triplets) / (i+1))

    return train_loss, total_tsm_loss, total_reg_loss, n_triplets, n_pos_triplets



@torch.no_grad()
def test(test_loader, device, model, aca_loss):
    model.eval()
    total_examples = 0
    total_loss = 0
    total_tsm_loss = 0
    total_reg_loss = 0
    n_triplets = []
    n_pos_triplets = []
    mse = []
    for i, data in enumerate(test_loader):
        data = data.to(device)
        predictions, embeddings = model(data.x.float(), data.edge_index,
                                        data.edge_attr, data.batch)
        loss_out = aca_loss(labels=data.y,
                            predictions=predictions,
                            embeddings=embeddings)

        loss, reg_loss, tsm_loss, n, n_pos = loss_out

        total_loss += float(loss) * data.num_graphs
        total_tsm_loss += float(tsm_loss) * data.num_graphs
        total_reg_loss += float(reg_loss) * data.num_graphs
        total_examples += data.num_graphs

        n_triplets.append(int(n))
        n_pos_triplets.append(int(n_pos))

        mse.append(F.mse_loss(predictions, data.y, reduction='none').cpu())

    test_loss = total_loss / total_examples
    total_tsm_loss = total_tsm_loss / total_examples
    total_reg_loss = total_reg_loss / total_examples
    n_triplets = int(sum(n_triplets) / (i+1))
    n_pos_triplets = int(sum(n_pos_triplets) / (i+1))
    
    test_rmse = float(torch.cat(mse, dim=0).mean().sqrt())
    
    return test_loss, total_tsm_loss, total_reg_loss, n_triplets, n_pos_triplets, test_rmse




## data HPs
n_repeats = 3
n_folds = 5

# trianing HPs
epochs = 2000
batch_size = 128
lr = 5e-5


weights = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #, 

cliffs = [(0.5, 1.5)]
names = ['braf']

for dataset_name, cliff in zip(names, cliffs):
    
    pre_transform = Gen39AtomFeatures()
    dataset = LSSNS(root = './tempignore/%s' % dataset_name, name=dataset_name, pre_transform=pre_transform)

    result_save_dir = './results/%s/' % dataset_name
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    # model HPs
    pub_args = {'in_channels':pre_transform.in_channels, 
                'edge_dim':pre_transform.edge_dim, 
                'batch_norms':None,
                'out_channels':1}


    cliff_lower, cliff_upper = cliff

    y = dataset.data.y.cpu().numpy()
    idx = np.argsort(y,axis=0).reshape(-1,) #sort data by ther value

    allres = []
    for alpha in tqdm(weights):
        # 5-fold cross-validation
        for j in range(n_repeats):
            for i in range(n_folds):
                ts = pd.Series(idx)
                ts_idx = ts.iloc[i::n_folds].tolist()
                tr_idx = ts[~ts.isin(ts_idx)].tolist()
                print(len(tr_idx), len(ts_idx))

                train_dataset = dataset.index_select(tr_idx)
                val_dataset = dataset.index_select(ts_idx)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                ## model HPs
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ACANet_PNA(**pub_args, aggregators=['mean', 'min', 'max', 'sum'],
                                   scalers=['identity','amplification', 'attenuation'],
                                   deg = get_deg(train_dataset)).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=10**-5)
                aca_loss = ACALoss(alpha=alpha, cliff_lower=cliff_lower, cliff_upper=cliff_upper)

                history = []
                for epoch in tqdm(range(1, epochs+1),ascii=True):
                    train_out = train(train_loader, device, optimizer, model, aca_loss)
                    val_out = test(val_loader, device, model, aca_loss)

                    train_loss, train_tsm, train_reg, train_n_triplets, train_n_pos_triplets = train_out
                    val_loss, val_tsm, val_reg, val_n_triplets, val_n_pos_triplets, val_rmse = val_out

                    history.append({'Epoch':epoch, 'train_loss':train_loss, 'train_triplet_loss':train_tsm,
                                    'train_mae_loss':train_reg, 'train_n_triplets': train_n_triplets, 
                                    'train_pos_triplets': train_n_pos_triplets,
                                    'val_loss':val_loss, 'val_triplet_loss':val_tsm,
                                    'val_mae_loss':val_reg, 'val_n_triplets':val_n_triplets, 
                                    'val_pos_triplets': val_n_pos_triplets,
                                    'val_rmse':val_rmse})

                df1 = pd.DataFrame(history)
                df1['fold'] = 'fold_%s' % (i+1)
                df1['repeat'] = 'repeat_%s' % (j+1)
                df1['cliff_lower'] = [cliff_lower for i in range(len(df1))]
                df1['cliff_upper'] = [cliff_upper for i in range(len(df1))]
                df1['alpha'] = [alpha for i in range(len(df1))]
                df1['dataset'] = dataset_name
                allres.append(df1)

    df = pd.concat(allres)
    df.to_pickle(os.path.join(result_save_dir, '%s_alpha_change_2000.pkl' % dataset_name))
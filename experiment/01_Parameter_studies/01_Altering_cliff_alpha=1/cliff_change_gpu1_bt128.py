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


gpuid = 0
torch.cuda.set_device(gpuid)


def train(train_loader, device, optimizer, model, aca_loss):
    total_examples = 0
    total_loss = 0
    total_tsm_loss = 0
    total_reg_loss = 0
    n_triplets = []
    n_pos_triplets = []
    model.train()
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
    total_examples = 0
    total_loss = 0
    total_tsm_loss = 0
    total_reg_loss = 0
    n_triplets = []
    n_pos_triplets = []
    mse = []
    model.eval()
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


for dataset_name in ['braf']: 

    pre_transform = Gen39AtomFeatures()
    dataset = LSSNS(root = './tempignore/%s' % dataset_name, 
                    name=dataset_name, 
                    pre_transform=pre_transform)

    result_save_dir = './results/%s/' % dataset_name

    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

    epochs = 1000
    batch_size = 128
    lr = 1e-4
    
    ## data HPs
    n_repeats = 1
    n_fold = 5
    
    ## loss HPs
    alpha = 1

    ## model HPs
    pub_args = {'in_channels':pre_transform.in_channels, 
                'edge_dim':pre_transform.edge_dim,  
                'out_channels':1,
                'batch_norms':None,
               }

    y = dataset.data.y.cpu().numpy()
    idx = np.argsort(y,axis=0).reshape(-1,) #sort data by ther value

    cliffs = list(np.arange(0.0, 4, 0.2).round(2)) #3.2
    low_up_cliffs = []
    for lower in cliffs:
        for upper in cliffs:
            if upper >= lower:
                low_up_cliffs.append((lower, upper))

    allres = []

    for j in range(n_repeats):
        for (cliff_lower, cliff_upper) in tqdm(low_up_cliffs,  ascii=True):
            for i in range(n_fold):
                ts = pd.Series(idx)
                ts_idx = ts.iloc[i::n_fold].tolist()
                tr_idx = ts[~ts.isin(ts_idx)].tolist()

                train_dataset = dataset.index_select(tr_idx)
                val_dataset = dataset.index_select(ts_idx)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                # model HPs
                device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')

                deg = get_deg(train_dataset)
                model = ACANet_PNA(**pub_args, aggregators=['mean', 'min', 'max', 'sum','std'],
                                   scalers=['identity','amplification','attenuation'],
                                   deg=deg).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(), lr=lr, weight_decay=10**-5)
                aca_loss = ACALoss(alpha=alpha, cliff_lower=cliff_lower, cliff_upper=cliff_upper)

                history = []
                for epoch in tqdm(range(epochs), desc = 'epoch', ascii=True):

                    train_loss, train_tsm, train_reg, train_n_triplets, train_n_pos_triplets = train(train_loader, device, optimizer, model, aca_loss)
                    val_loss, val_tsm, val_reg, val_n_triplets, val_n_pos_triplets, val_rmse = test(val_loader, device, model, aca_loss)

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
                df1['cliff'] = [(cliff_lower, cliff_upper) for i in range(len(df1))]
                df1['alpha'] = [alpha for i in range(len(df1))]
                df1['dataset'] = dataset_name
                allres.append(df1)


    df = pd.concat(allres)
    save_file = os.path.join(result_save_dir, '%s.pkl' % dataset_name)
    df.to_pickle(save_file)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from math import sqrt
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy.stats import pearsonr

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn.models import AttentiveFP

# import clsar package
import sys, os
sys.path.insert(0, '/home/shenwanxiang/Research/bidd-clsar/')
from clsar.dataset import LSSNS # dataset
from clsar.feature import Gen115AtomFeatures, GenAttentiveFeatures # feature
from clsar.model.model import ACNet_GCN, ACNet_GIN, ACNet_GAT, ACNet_PNA # model
from clsar.model.loss import ada_batch_all_triplet_loss, ada_batch_all_triplet_loss2


def train(model, train_loader, device, cliff, alpha, reg_mse):
    total_examples = 0
    total_loss =  0    
    total_triplet_loss = 0
    total_mae_loss = 0   
    n_triplets = []
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        predictions, embeddings = model(data.x.float(), data.edge_index, 
                                        data.edge_attr, data.batch)
        loss_out = ada_batch_all_triplet_loss(embeddings = embeddings,
                                              predictions = predictions,
                                              labels = data.y, 
                                              device = device, 
                                              cliff = cliff,
                                              alpha = alpha,
                                              reg_mse = reg_mse
                                              )
        loss, triplet_loss, mae_loss, num_positive_triplets = loss_out

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_triplet_loss += float(triplet_loss) * data.num_graphs        
        total_mae_loss += float(mae_loss) * data.num_graphs        
        total_examples += data.num_graphs
        n_triplets.append(int(num_positive_triplets))

    train_loss = total_loss / total_examples
    total_triplet_loss = total_triplet_loss / total_examples
    total_mae_loss = total_mae_loss / total_examples
    n_triplets = int(sum(n_triplets) / (i+1))

    return train_loss, total_triplet_loss, total_mae_loss, n_triplets


@torch.no_grad()
def test(model, loader, device, cliff, alpha, reg_mse):
    
    total_examples = 0
    total_loss =  0    
    total_triplet_loss = 0
    total_mae_loss = 0   
    n_triplets = []
    mse = []
    ps = []
    ts = []
    for i, data in enumerate(val_loader):
        data = data.to(device)
        predictions, embeddings = model(data.x.float(), data.edge_index, data.edge_attr, data.batch)
        loss_out = ada_batch_all_triplet_loss(embeddings = embeddings,
                                              predictions = predictions,
                                              labels = data.y, 
                                              device = device, 
                                              cliff = cliff, 
                                              alpha=alpha, reg_mse = reg_mse)
        loss, triplet_loss, mae_loss, num_positive_triplets = loss_out
        
        total_loss += float(loss) * data.num_graphs
        total_triplet_loss += float(triplet_loss) * data.num_graphs        
        total_mae_loss += float(mae_loss) * data.num_graphs        
        total_examples += data.num_graphs
        n_triplets.append(int(num_positive_triplets))
        mse.append(F.mse_loss(predictions, data.y, reduction='none').cpu())
        
        p = predictions.cpu().detach().numpy().reshape(-1,)
        t =  data.y.cpu().detach().numpy().reshape(-1,)
        ps.append(p)
        ts.append(t)
        
    total_loss = total_loss / total_examples
    total_triplet_loss = total_triplet_loss / total_examples
    total_mae_loss = total_mae_loss / total_examples
    n_triplets = int(sum(n_triplets) / (i+1))
    rmse = float(torch.cat(mse, dim=0).mean().sqrt()) 
    
    ps = np.concatenate(ps)
    ts = np.concatenate(ts)
    pcc, p_value = pearsonr(ts, ps)
    r2 = pcc**2
    return total_loss, total_triplet_loss, total_mae_loss, rmse, r2, n_triplets




Dataset =  LSSNS # MoleculeNet
result_save_dir = './results/cliff_with_baseline/'
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

epochs = 1000
batch_size = 128

## data HPs
cliffs = ['mae', 'mse']
cls = list(np.arange(0, 4.0, 0.2).round(2))
cliffs.extend(cls)

n_repeats = 10
n_fold = 5
## model HPs
pub_args = {'in_channels':115, 'hidden_channels':64, 'out_channels':1, 
            'edge_dim':10, 'num_layers':10, 'dropout_p':0.1, 'batch_norms':None}

for dataset_name in list(Dataset.names.keys())[:1]:
    #dataset_name = 'eaat3'
    dataset = Dataset(root = './tmpignore/', name=dataset_name, pre_transform=Gen115AtomFeatures())
    y = dataset.data.y.cpu().numpy()
    idx = np.argsort(y,axis=0).reshape(-1,) #sort data by ther value

    allres = []
    for cliff in cliffs:
        # n-fold cross-validation
        if type(cliff) == str:
            if cliff == 'mse':
                reg_mse = True
                cliff_use = 1e10
            else:
                reg_mse = False
                cliff_use = 1e10
            alpha = 0.0
            
        else:
            alpha = 1.0
            reg_mse = False
            cliff_use = cliff
            
        for j in range(n_repeats):
            for i in range(n_fold):
                ts = pd.Series(idx)
                ts_idx = ts.iloc[i::n_fold].tolist()
                tr_idx = ts[~ts.isin(ts_idx)].tolist()
                print(len(tr_idx), len(ts_idx))

                train_dataset = dataset.index_select(tr_idx)
                val_dataset = dataset.index_select(ts_idx)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                ## model HPs
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = ACNet_GAT(**pub_args, heads = 3, dropout= 0.1).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=10**-3.5,
                                             weight_decay=10**-5)
                history = []

                for epoch in range(1, epochs):
                    train_loss, triplet_loss, mae_loss, n_triplets = train(model, train_loader, device, 
                                                                           cliff_use, alpha, reg_mse)
                    val_out = test(model, val_loader, device, cliff_use, alpha, reg_mse)
                    val_loss, val_triplet_loss, val_mae_loss, val_rmse, val_r2, val_n_triplets = val_out

                    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f} TripLoss: {triplet_loss:.4f} MAELoss: {mae_loss:.4f} '
                          f'Triplets: {n_triplets:03d}; Val: {val_rmse:.4f}')

                    history.append({'Epoch':epoch, 'train_loss':train_loss, 'train_triplet_loss':triplet_loss,
                                    'train_mae_loss':mae_loss, 'train_triplets': n_triplets,
                                    'val_loss':val_loss, 'val_triplet_loss':val_triplet_loss,
                                    'val_mae_loss':val_mae_loss, 'val_n_triplets':val_n_triplets, 
                                    'val_rmse':val_rmse, 'val_r2':val_r2})

                df1 = pd.DataFrame(history)
                df1['fold'] = 'fold_%s' % (i+1)
                df1['repeat'] = 'fold_%s' % (j+1)
                df1['cliff'] = [cliff for i in range(len(df1))]
                #df1['alpha'] = [alpha for i in range(len(df1))]
                #df1['batch_size'] = [batch_size for i in range(len(df1))]
                #df1['model_paras'] = [pub_args for i in range(len(df1))]
                df1['dataset'] = dataset_name
                allres.append(df1)

    df = pd.concat(allres)
    save_file = os.path.join(result_save_dir, '%s.pkl' % dataset_name)
    df.to_pickle(save_file)
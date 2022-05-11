import torch
import json
import os.path as osp
from math import sqrt
import numpy as np

import torch
import torch.nn.functional as F
from rdkit import Chem

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn.models import AttentiveFP

import os
import os.path as osp
import re

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn.models import AttentiveFP

from model import GNN,GNN_graphpred
from dataset import LSSInhibitor,GenAtomFeatures,GenAttentiveFeatures
from loss import ada_batch_all_triplet_loss
from splitters import scaffold_split, random_split, random_scaffold_split, cv_random_split


device = 'cuda:4'
batch_size = 32
epochs = 200
lr = 0.0001
decay = 1e-2
num_layer = 5
emb_dim = 300
dropout_ratio = 0
JK = 'last'
dataset = 'EAAT3'
output_model_file = ''
gnn_type = 'gin'
seed = 0
num_workers = 8
mode = 'ada_batch_all_triplet_loss'   #ada_batch_all_triplet_loss,ada_batch_hard_triplet_loss,triplet_loss
feature_type = 'custom'  #random,onehot,custom,pseudo
graph_pooling = 'set2set2' #mean,last,sum,set2set,atention
cliff_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
#cliff_list = [0.1]
def train():
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        '''
        data.x = data.x.to(torch.float32)
        data.edge_index=data.edge_index.to(torch.long)
        data.edge_attr=data.edge_attr.to(torch.float32)
        data.batch=data.batch.to(torch.long)
        '''
        #out = model(float(data.x), data.edge_index.to(torch.long), data.edge_attr.to(torch.float32), data.batch.to(torch.long))
        emb,pre = model(data.x, data.edge_index, data.edge_attr, data.batch)

        #loss = F.mse_loss(out, data.y)
        loss = ada_batch_all_triplet_loss(embeddings=emb, labels=data.y, prediction=pre, device=device, minv=minv, maxv=maxv, weight=0.5, cliff=cliff,squared=False)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return sqrt(total_loss / total_examples)


@torch.no_grad()
def test(loader):
    mse = []
    for data in loader:
        data = data.to(device)
        emb,pre = model(data.x.to(torch.float32), data.edge_index, data.edge_attr, data.batch)
        pre = pre*(maxv-minv)+minv
        mse.append(F.mse_loss(pre, data.y, reduction='none').cpu())
        #print('mse:',mse)
    return float(torch.cat(mse, dim=0).mean().sqrt())


if __name__ == "__main__":
    best_allcliff_allset = {}
    for cliff in cliff_list:
        best_result_allset = []
        for dataset_name in LSSInhibitor.names.keys():
            if dataset_name == 'notum':
                print(dataset_name + "\n")
                path = './data'

                # use the attentiveFP node and edge features during the mol-2-graph transoformation
                dataset = LSSInhibitor(path, name=dataset_name, pre_transform=GenAtomFeatures('custom')).shuffle()
                #dataset = LSSInhibitor(path, name=dataset_name, pre_transform=GenAttentiveFeatures()).shuffle()
                #dataset = MoleculeNet(path, name='FreeSolv', pre_transform=GenFeatures()).shuffle()
                print(dataset.__dict__)

                #get min&max for normalization
                minv = 1e12
                maxv = -1e12
                for i in dataset:
                    if i.y<minv:
                        minv = i.y.item()
                    if i.y>maxv:
                        maxv = i.y.item()
                print(f'minv:{minv}  maxv:{maxv}') 

                #split by values to 5 folds
                sorted_dataset = sorted(dataset,key = lambda data:data.y )
                split_dict={}
                for i in range(5):
                    split_dict[i] = {'train':[],'test':[]}
                for i in range(len(sorted_dataset)):
                    tmp = i % 5
                    for j in range(5):
                        if tmp == j:
                            split_dict[j]['test'].append(sorted_dataset[i])
                        else:
                            split_dict[j]['train'].append(sorted_dataset[i])
                print(split_dict)
                            
                '''
                N = len(dataset) // 5
                val_dataset = dataset[:N]
                test_dataset = dataset[N:2 * N]
                train_dataset = dataset[2 * N:]
                '''
                best_result = []

                for fold in range(len(split_dict)):

                    train_dataset = split_dict[fold]['train']
                    test_dataset = split_dict[fold]['test']
                    #exit(0)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)

                    device = torch.device(device if torch.cuda.is_available() else 'cpu')

                    model = GNN_graphpred(num_layer, emb_dim, num_tasks = 1, JK = JK, drop_ratio =dropout_ratio,feature_type = feature_type,
                                          graph_pooling = graph_pooling, gnn_type = gnn_type).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                                 weight_decay=decay)


                    test_best = 1e3
                    for epoch in range(epochs):
                        train_rmse = train()
                        test_rmse = test(test_loader)
                        if test_rmse<test_best:
                            test_best = test_rmse

                        print(f'Epoch: {epoch:03d}, Dataset:{dataset_name} Cliff:{cliff} Fold:{fold} Loss: {train_rmse:.4f} '
                              f'Test: {test_rmse:.4f} Test_best:{test_best:.4f}')
                    best_result.append(test_best)
                print(f'datast_name:{dataset_name} BestResultOneSet:{best_result}')
                best_result_allset.append(best_result)
        print('BestResultAllSet:',best_result_allset)
        best_allcliff_allset[cliff]= best_result_allset
    print(best_allcliff_allset)
    with open("best_allcliff_allset.txt","w") as f:
        f.write(str(best_allcliff_allset))
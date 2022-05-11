import torch
import json
import os.path as osp
from math import sqrt
import numpy as np
import random
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
#from loss import ada_batch_all_triplet_loss
from splitters import scaffold_split, random_split, random_scaffold_split, cv_random_split
'''
def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda()
    else:
        return module.cpu()
 '''   
def cudafy(module):
    return module.cpu()
def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr
def _calc_ecfp4(smiles):
    ecfp4 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), radius = 2)    
    return ecfp4
def _calc_ecfp4_hash(smiles):
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles),radius =2,nBits=1024)
    return ecfp4
def pairwise_distance(embeddings, squared=True):
    pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                 torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                 2.0 * torch.matmul(embeddings, embeddings.t())

    error_mask = pairwise_distances_squared <= 0.0
    if squared:
        pairwise_distances = pairwise_distances_squared.clamp(min=0)
    else:
        pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

    pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

    num_data = embeddings.shape[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(cudafy(torch.ones([num_data])))
    #mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(torch.ones([num_data]))
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances
def get_triplet_mask(labels, cliff, device):

    indices_equal = torch.eye(labels.shape[0]).bool()
    indices_not_equal = torch.logical_not(indices_equal)
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k).to(device)

    #labels = torch.unsqueeze(labels, -1)
    #print('labels:',labels)
    target_l1_dist = torch.cdist(labels,labels,p=1) 
    label_equal = target_l1_dist < cliff
    #print('label_equal:',label_equal)
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k)).to(device)
    #print('val_indice',valid_labels[0])

    mask = torch.logical_and(distinct_indices, valid_labels)
    return mask             
def ada_batch_all_triplet_loss(embeddings, labels, prediction, device, minv, maxv, cliff=0.5, weight=0.5, squared=False):
    '''
       union loss of a batch
       -------------------------------
       Args:
          labels:     shape = （batch_size,）
          embeddings: 提取的特征向量， shape = (batch_size, vector_size)
          margin:     margin大小， scalar

       Returns:
         union_loss: scalar, 一个batch的损失值
    '''

    # 得到每两两embeddings的距离，然后增加一个维度，一维需要得到（batch_size, batch_size, batch_size）大小的3D矩阵
    # 然后再点乘上valid 的 mask即可
    nor_labels = (labels-minv)/(maxv-minv)   
    labels_dist = (nor_labels - nor_labels.T).abs()  

    margin_pos =  labels_dist.unsqueeze(2)
    margin_neg =  labels_dist.unsqueeze(1)
    margin = margin_neg - margin_pos

    pairwise_dis = pairwise_distance(embeddings=embeddings, squared=squared)
    anchor_positive_dist = pairwise_dis.unsqueeze(2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    anchor_negative_dist = pairwise_dis.unsqueeze(1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)
    #triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(labels=labels, cliff = cliff, device=device )
    mask = mask.float()

    triplet_loss = torch.mul(mask, triplet_loss)
    triplet_loss = torch.maximum(triplet_loss, torch.tensor([0.0]).to(device))
    # 计算valid的triplet的个数，然后对所有的triplet loss求平均
    valid_triplets = (triplet_loss> 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)

    #fraction_postive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    #triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
    triplet_loss = torch.sum(triplet_loss) / (num_valid_triplets + 1e-16)
    #print('三元组损失：',triplet_loss)
    #print('fraction',fraction_postive_triplets)
    #pre_loss = torch.mean((nor_labels-prediction).abs())
    pre_loss = torch.mean((nor_labels-prediction).abs())
    union_loss = 1*triplet_loss+ 0.7*pre_loss
    return union_loss

device = 'cpu'
batch_size = 16
epochs = 200
lr = 0.0001
decay = 1e-3
num_layer = 5
emb_dim = 300
dropout_ratio = 0
JK = 'last'
dataset = 'EAAT3'
output_model_file = ''
gnn_type = 'gin'
seed = 88
num_workers = 8
mode = 'ada_batch_all_triplet_loss'   #ada_batch_all_triplet_loss,ada_batch_hard_triplet_loss,triplet_loss
feature_type = 'custom'  #random,onehot,custom,pseudo
graph_pooling = 'set2set2' #mean,last,sum,set2set,atention
cliff_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
#cliff_list = [ 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    best_allcliff_allset = {}
    mean_allcliff_allset = {}
    for cliff in cliff_list:
        best_result_allset = []
        mean_result_allset = []
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
                mean_result = []
                for fold in range(len(split_dict)):

                    train_dataset = split_dict[fold]['train']
                    test_dataset = split_dict[fold]['test']
                    #exit(0)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

                    device = torch.device(device if torch.cuda.is_available() else 'cpu')

                    model = GNN_graphpred(num_layer, emb_dim, num_tasks = 1, JK = JK, drop_ratio =dropout_ratio,feature_type = feature_type,
                                          graph_pooling = graph_pooling, gnn_type = gnn_type).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                                 weight_decay=decay)

                    conv_result = []
                    test_best = 1e3
                    for epoch in range(epochs):
                        train_rmse = train()
                        test_rmse = test(test_loader)
                        if test_rmse<test_best:
                            test_best = test_rmse
                        if epoch >= 150:
                            conv_result.append(test_rmse)
                        print(f'Epoch: {epoch:03d}, Dataset:{dataset_name} Cliff:{cliff} Fold:{fold} Loss: {train_rmse:.4f} '
                              f'Test: {test_rmse:.4f} Test_best:{test_best:.4f}')
                    best_result.append(test_best)
                    mean_result.append(np.array(conv_result).mean())
                print(f'datast_name:{dataset_name} BestResultOneSet:{best_result} MeanResultOneSet:{mean_result}')
                best_result_allset.append(best_result)
                mean_result_allset.append(mean_result)
        print('BestResultAllSet:',best_result_allset)
        best_allcliff_allset[cliff]= best_result_allset
        mean_allcliff_allset[cliff]= mean_result_allset
    print('BEST:',best_allcliff_allset)
    print('MEAN::',mean_allcliff_allset)
    with open("best_allcliff_usp7.txt","w") as f:
        f.write(str(best_allcliff_allset))
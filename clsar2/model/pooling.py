# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:44:05 2023

@author: wanxiang.shen
"""
import torch
from torch_scatter import scatter
from torch_scatter.utils import broadcast

class SubstructurePool:
    
    def __init__(self, reduce = 'sum'):
        '''
        reduce: sum, mean, max, min
        '''
        self.reduce = reduce
        self.__name__ = 'SubstructurePool'
        
        
    def __call__(self, x, batch, fp):
        return local_substructure_pool(x, batch, fp, self.reduce)


    
def local_substructure_pool(x, batch, fp, reduce = 'sum'):

    '''
    LSP: Returns batch-wise subgraph-level-output, 3D tensor, (batch_size, fingerprint_dim, in_channel), 
    if in_channel is 1, batch size is 32, fingerprint dim is 1024, then the output shape is : (32, 1024, 1)
    
     Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        fp (BoolTensor, optional): node atom fingerprint matrix (MxN, N: 1024, M: 312, number of nodes).
            The subgraph info for each sample
        reduce: `sum` (`add`)`max`, `mean` and `min`..
    '''
    ## in_channel = 1
    atom_size, in_channel = x.shape
    sample_size = int(batch.max().item() + 1)
    fp = fp.to(batch.dtype)
    
    ## sub-structure pooling for each channel
    all_channel_pool = [] 
    for i in range(in_channel): # per-channel pooling, in_channel equals 1 in our GNN model
        x1 = x[:, i] #
        all_mol_pool = []
        for s in range(sample_size): # per-mol pooling
            sample_mask = batch == s
            mol_fp = fp[sample_mask]
            mol_x = x1[sample_mask]
            mol_x = broadcast(mol_x, mol_fp, dim=-2)
            mol_pool_out = scatter(mol_x,  mol_fp,  dim=-2, dim_size=2, reduce=reduce)
            offbit, onbit = mol_pool_out #{0 & 1, same as batch 0,1,2,3,4}, we only need to collect 1.        
            all_mol_pool.append(onbit)
        one_channel_pool = torch.stack(all_mol_pool)
        all_channel_pool.append(one_channel_pool)
            
    #3D tensor, (batch_size, fingerprint_dim, in_channel)
    substructure_pool = torch.stack(all_channel_pool, axis=-1) 
    ## if in_channel == 1: (32, 1024, 1), you can squeeze the dim into (32, 1024):
    #substructure_pool_res.squeeze(dim=-1)
    return substructure_pool



def _local_substructure_pool(x, batch, fp, reduce = 'sum'):


    '''
    LSP: Returns batch-wise subgraph-level-output, 3D tensor, (batch_size, fingerprint_dim, in_channel), 
    if in_channel is 1, batch size is 32, fingerprint dim is 1024, then the output shape is : (32, 1024, 1)
    
     Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        fp (BoolTensor, optional): node atom fingerprint matrix (MxN, N: 1024, M: 312, number of nodes).
            The subgraph info for each sample
        reduce: `sum` (`add`) or `max`.
            Please note: `mean` and `min` are not suitable in this function,
            Rewrite pool if you want the "mean" and "min" operation.
    '''

    num_atom, in_channel = x.shape
    size = int(batch.max().item() + 1)
    fp = fp.to(x.dtype)
    
    ## sub-structure pooling for each channel
    per_channel_pool_res = []
    for i in range(in_channel):
        x1 = x[:, i] #
        x1 = broadcast(x1, fp, dim=-2)
        x1 = torch.mul(x1, fp)
        pool_out = scatter(x1, batch, dim=-2, dim_size=size, reduce=reduce)
        per_channel_pool_res.append(pool_out)

    #3D tensor, (batch_size, fingerprint_dim, in_channel)
    substructure_pool_res = torch.stack(per_channel_pool_res, axis=-1) 

    ## if in_channel == 1: (32, 1024, 1), you can squeeze the dim into (32, 1024):
    #substructure_pool_res.squeeze(dim=-1)
    
    return substructure_pool_res
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:34:56 2022

@author: wanxiang.shen@u.nus.edu

The below models support message passing with multi-dimensional edge feature information, i.e., the edge_attr data
For more details, pls refer to this link: https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html
Good video to watch:https://www.youtube.com/watch?v=mdWQYYapvR8
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ModuleList, ReLU, Dropout

from torch_geometric.nn import MessagePassing, JumpingKnowledge
from torch_geometric.nn import NNConv, GATv2Conv, PNAConv, SAGEConv, GINEConv, MLP 
from torch_geometric.nn import global_mean_pool, global_max_pool, Set2Set, GlobalAttention
from torch_geometric.utils import degree
from copy import deepcopy 




def _fix_reproducibility(seed=42):
    import os, random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

class ACANet_Base(torch.nn.Module):

    
    r"""An base class for implementing activlity cliff graph neural networks (ACNets) 
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        convs_layers: Message passing layers. (default: :[64, 128, 256, 512])
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj: torch_geometric.nn.global_max_pool) 
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 edge_dim,
                 convs_layers = [64, 128, 256, 512], #[64, 512, 1024],  
                 dense_layers = [256, 128, 32], #
                 pooling_layer = global_max_pool,
                 batch_norms = torch.nn.BatchNorm1d,
                 dropout_p = 0.0,
                 **kwargs,
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim

        self.convs_layers = convs_layers
        self.dropout_p = dropout_p
 
        self.pooling_layer = pooling_layer
        self.dense_layers = dense_layers
        self._batch_norms = batch_norms

        ## convs stack 
        _convs_layers = [in_channels]
        _convs_layers.extend(convs_layers)
        self._convs_layers = _convs_layers
        self.convs = ModuleList()
        for i in range(len(_convs_layers)-1):
            convs = self.init_conv(_convs_layers[i], _convs_layers[i+1], edge_dim, **kwargs)
            self.convs.append(convs)

        # norm stack
        self.batch_norms = None
        if batch_norms is not None:
            self.batch_norms = ModuleList()
            for i in range(len(_convs_layers)-1):
                self.batch_norms.append(deepcopy(batch_norms(_convs_layers[i+1])))


        _dense_layers = [_convs_layers[-1]]
        _dense_layers.extend(dense_layers)
        self._dense_layers = _dense_layers
        
        self.lins = ModuleList()
        for i in range(len(_dense_layers)-1):
            lin = Linear(_dense_layers[i], _dense_layers[i+1])
            self.lins.append(lin)

        
        # Output layer
        last_hidden = _dense_layers[-1]
        self.out = Linear(last_hidden, out_channels)

        model_args = {'in_channels':self.in_channels, 
                'out_channels':self.out_channels,
                'edge_dim':self.edge_dim, 
                'convs_layers': self.convs_layers, 
                'dropout_p':self.dropout_p, 
                'batch_norms':self._batch_norms,
                'pooling_layer':self.pooling_layer,
                'dense_layers':self.dense_layers,
               }
        for k, v in kwargs.items():
            model_args[k] = v
            setattr(self, k, v)   
            
        self.model_args = model_args


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.batch_norms or []:
            norm.reset_parameters()
        if hasattr(self, 'out'):
            self.out.reset_parameters()          
  
        
    def init_conv(self, in_channels, out_channels,  **kwargs):
        raise NotImplementedError
        

    def forward(self, x, edge_index, edge_attr, batch,  *args, **kwargs):
        '''
        data.x, data.edge_index, data.edge_attr, data.batch, data.fp,...
        '''
        
        x = F.dropout(x, p=self.dropout_p, training = self.training)
        # conv-act-norm-drop layer
        for i, convs in enumerate(self.convs):
            x = convs(x, edge_index, edge_attr, *args, **kwargs)        
            x = F.relu(x, inplace=True)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # global pooling layer: pooling on the whole molecular structure
        embed = self.pooling_layer(x, batch)
        y = self.pooling_layer(x, batch)

        # dense layer
        for lin in self.lins:
            y = F.relu(lin(y), inplace=True)
            y = F.dropout(y, p=self.dropout_p, training=self.training)

        #output layer
        y = self.out(y)
        return y, embed
    
    
    
class ACANet_GCN(ACANet_Base):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper, using the
    :class:`~torch_geometric.nn.conv.NNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        convs_layers: Message passing layers. (default: :[64, 512, 1024])
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj: torch_geometric.nn.global_max_pool) 
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    
    # integrate multi-dimensional edge features, GCNConv only accepts 1-d edge features
    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs):
        
        # To map each edge feature to a vector of shape (in_channels * out_channels) as weight to compute messages.
        edge_fuc = Sequential(Linear(edge_dim, in_channels * out_channels))
        return NNConv(in_channels, out_channels, nn = edge_fuc, **kwargs)

    
class ACANet_GIN(ACANet_Base):

    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper, using the :class:`~torch_geometric.nn.GINEConv` operator for message passing.
    It is able to corporate edge features into the aggregation procedure. 

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        convs_layers: Message passing layers. (default: :[64, 512, 1024])
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj: torch_geometric.nn.global_max_pool) 
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs): 
        #A neural network :math:`h_{\mathbf{\Theta}}` that maps node features `x` of shape `[-1, in_channels]` to `[-1, out_channels]`
        node_fuc = Sequential(Linear(in_channels, out_channels))
        return GINEConv(nn = node_fuc, edge_dim = edge_dim, **kwargs)
    
    
    
class ACANet_GAT(ACANet_Base):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        convs_layers: Message passing layers. (default: :[64, 512, 1024])
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj: torch_geometric.nn.global_max_pool) 
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    
    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs): 
        #False concat the head, to average the information
        concat = kwargs.pop('concat', False)
        return GATv2Conv(in_channels, out_channels, edge_dim = edge_dim, concat=concat, **kwargs)

    
    
    
class ACANet_PNA(ACANet_Base):
    r"""The Graph Neural Network from the `"Principal Neighbourhood Aggregation
    for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
    :class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality.
        convs_layers: Message passing layers. (default: :[64, 512, 1024])
        dense_layers: Fully-connected layers. (default: :[512, 128, 32])
        pool_layer: (torch_geometric.nn.Module, optional): the pooling-layer. (default: :obj: torch_geometric.nn.global_max_pool) 
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        aggregators(list of str): Set of aggregation function identifiers, e.g., ['mean', 'min', 'max', 'sum']
        scalers(list of str): Set of scaling function identifiers, e.g., ['identity', 'amplification', 'attenuation'] 
        deg (Tensor): Histogram of in-degrees of nodes in the training set, used by scalers to normalize, e.g.,  torch.tensor([1, 2, 3]
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs): 
        return PNAConv(in_channels, out_channels, edge_dim = edge_dim,  **kwargs)

                

def get_deg(train_dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg





__all__ = ['ACANet_GCN', 'ACANet_GIN', 'ACANet_GAT', 'ACANet_PNA', 'get_deg']
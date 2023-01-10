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
from torch.nn import Sequential, Linear, ModuleList, ReLU

from torch_geometric.nn import MessagePassing, JumpingKnowledge
from torch_geometric.nn import NNConv, GATv2Conv, PNAConv, SAGEConv, GINEConv, MLP 
from torch_geometric.nn import global_mean_pool, global_max_pool, Set2Set, GlobalAttention
from torch_geometric.utils import degree


class ACANet_Base(torch.nn.Module):

    r"""An base class for implementing activlity cliff graph neural networks (ACNets) 
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality. 
        hidden_channels (int, optional): Size of each hidden sample. (default: :int:64)
        num_layers (int, optional): Number of message passing layers. (default: :int:2)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        global_pool: (torch_geometric.nn.Module, optional): the global-pooling-layer. (default: :obj: torch_geometric.nn.global_mean_pool)
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 edge_dim,
                 hidden_channels = 64, 
                 num_layers = 2,
                 dropout_p = 0.1,
                 batch_norms = None,
                 global_pool = global_mean_pool,
                 **kwargs,
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.jk_mode = 'cat'
        self.global_pool = global_pool
        
        ## layer stack 
        self.convs = ModuleList()
        self.convs.append(self.init_conv(in_channels, hidden_channels, edge_dim, **kwargs))
        for _ in range(num_layers - 1):
            self.convs.append(self.init_conv(hidden_channels, hidden_channels, edge_dim, **kwargs))
        
        # norm stack
        self.batch_norms = None
        if batch_norms is not None:
            self.batch_norms = ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(copy.deepcopy(batch_norms))

        self.jk = JumpingKnowledge(self.jk_mode, hidden_channels, num_layers)
        self.lin = Linear(num_layers * hidden_channels, self.out_channels)
      
        model_args = {'in_channels':self.in_channels, 
                'hidden_channels':self.hidden_channels, 
                'out_channels':self.out_channels,
                'edge_dim':self.edge_dim, 
                'num_layers': self.num_layers, 
                'dropout_p':self.dropout_p, 
                'batch_norms':self.batch_norms,
                'global_pool':self.global_pool
               }
        for k, v in kwargs.items():
            model_args[k] = v
            setattr(self, k, v)   
            
        self.model_args = model_args



    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.batch_norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()
            

    def init_conv(self, in_channels, out_channels,  **kwargs):
        raise NotImplementedError
        
        
        
    def forward(self, x, edge_index, edge_attr, batch,  *args, **kwargs):

        x = F.dropout(x, p=self.dropout_p, training = self.training)
        
        # conv-act-norm-drop layer
        xs = []  
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr, *args, **kwargs)        
            x = F.relu(x, inplace=True)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            xs.append(x) #for jk
            
        # the jk layer        
        x = self.jk(xs)
        
        # global pooling layer, please replace it with fuctinal group pooling @cuichao
        embed = self.global_pool(x, batch)
        
        # output
        y = self.lin(embed)
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
        hidden_channels (int, optional): Size of each hidden sample. (default: :int:64)
        num_layers (int, optional): Number of message passing layers. (default: :int:2)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        global_pool: (torch_geometric.nn.Module, optional): the global-pooling-layer. (default: :obj: torch_geometric.nn.global_mean_pool)
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
        hidden_channels (int, optional): Size of each hidden sample. (default: :int:64)
        num_layers (int, optional): Number of message passing layers. (default: :int:2)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        global_pool: (torch_geometric.nn.Module, optional): the global-pooling-layer. (default: :obj: torch_geometric.nn.global_mean_pool)
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
        hidden_channels (int, optional): Size of each hidden sample. (default: :int:64)
        num_layers (int, optional): Number of message passing layers. (default: :int:2)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        global_pool: (torch_geometric.nn.Module, optional): the global-pooling-layer. (default: :obj: torch_geometric.nn.global_mean_pool)
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
        hidden_channels (int, optional): Size of each hidden sample. (default: :int:64)
        num_layers (int, optional): Number of message passing layers. (default: :int:2)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        global_pool(torch_geometric.nn.Module, optional): the global-pooling-layer. (default: :obj: torch_geometric.nn.global_mean_pool)
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



__all__ = ['ACANet_GCN', 'ACANet_GIN', 'ACANet_GAT', 'ACANet_PNA']
import argparse
import numpy as np

from torch_sparse import SparseTensor
from torch import Tensor
from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
#from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
import matplotlib.cm as cm

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from copy import deepcopy

from torch_scatter import scatter_add
from torch_geometric.utils import softmax




visual_dict = {}
visual_weight = []
visual_smiles = []

class Set2Set(torch.nn.Module):
    r"""The global pooling operator based on iterative content-based attention
    from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Args:
        in_channels (int): Size of each input sample.
        processing_steps (int): Number of iterations :math:`T`.
        num_layers (int, optional): Number of recurrent layers, *.e.g*, setting
            :obj:`num_layers=2` would mean stacking two LSTMs together to form
            a stacked LSTM, with the second LSTM taking in outputs of the first
            LSTM and computing the final results. (default: :obj:`1`)
    """

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x, batch):
        """"""
        batch_size = batch.max().item() + 1

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            weight = []
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)
            a = softmax(e, batch, num_nodes=batch_size)
            #print(a)
            for j in a:
                weight.append(j.item())
            visual_weight.append(weight)
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda()
    else:
        return module.cpu()

def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        #self.pool = global_mean_pool
        self.pool = Set2Set(300,1)
        #self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))
        self.projection_head = nn.Sequential(nn.Linear(600, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, y, batch):

        x = self.gnn(x, edge_index.to(torch.long), edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x, y

    def loss_unsup(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss
    def loss_supcl(self, x, y, contrast_mode: str = 'all'):

        '''
        #看x的维度，如果做增强的话加上这段，最好统一
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        '''
        temperature = 0.6
        base_temperature = 0.7
        print('y:',y)

        x = torch.unsqueeze(x, 1) # 增加一个view维度，本来应该输入x为3维但需要改gnn，也可以再改一下loss适配2维，如果将来用到增强，有监督gnn输出
        #的x肯定还是要用这个loss，无监督是loss_unsup
        print('x:',x)
        y = torch.unsqueeze(y, 0)

        print('X,shape',x.shape)
        batch_size = x.shape[0]
        mask = torch.eq(y, y.T).float()
        print('mask:',mask)
        contrast_count = x.shape[1] #对比的数目为view个数？等于增强个数？如果分子不增强那就一个？
        contrast_feature = torch.cat(torch.unbind(x, dim=1), dim=0) #取消view的维度，那batch内各个样本相同view先聚集在一起
        # ，再元组dim=0的cat
        print('contrast_feature',contrast_feature)
        print(contrast_feature.shape)

        if contrast_mode == 'one':
            anchor_feature = x[:, 0] #只用每个样本的第一个增强(第一个view)的特征作为anchor，【batch_size, feature_dim】
            anchor_count = 1
        elif contrast_mode == 'all':
            anchor_feature = contrast_feature #见58行，所有view作为anchor
            anchor_count = contrast_count #对比数就为view的个数
        else:
            raise ValueError('Unknown mode: {}'.format(contrast_mode))
        print('anchor_feature:', anchor_feature)
        print(anchor_feature.shape)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),#注意anchor_feature的取值
            temperature)
        print('anchor_dot_contrast',anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)#1.anchor是1即一种view下，【batch_size，batch_size*n_view】
        #2.anchor等于contrastcount
        print(anchor_count,contrast_count)
        print('MASK:', mask)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            #torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            torch.arange(batch_size * anchor_count).view(-1, 1),
            0
        )
        print('logits_mask', logits_mask)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        print('log_prob:', log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (temperature / base_temperature) * mean_log_prob_pos
        print('loss:', loss)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    '''
    三元semi-hard损失
    '''

    def masked_maximum(self, data, mask, dim=1):
        """Computes the axis wise maximum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the maximum.
            Returns:
              masked_maximums: N-D `Tensor`.
                The maximized dimension is of size 1 after the operation.
            """
        axis_minimums = torch.min(data, dim, keepdim=True).values
        masked_maximums = torch.max(torch.mul(data - axis_minimums, mask), dim, keepdim=True).values + axis_minimums
        return masked_maximums

    def masked_minimum(self, data, mask, dim=1):
        """Computes the axis wise minimum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the minimum.
            Returns:
              masked_minimums: N-D `Tensor`.
                The minimized dimension is of size 1 after the operation.
            """
        axis_maximums = torch.max(data, dim, keepdim=True).values
        masked_minimums = torch.min(torch.mul(data - axis_maximums, mask), dim, keepdim=True).values + axis_maximums
        return masked_minimums

    def pairwise_distance(self, embeddings, squared=True):
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

    def loss_semihard(self, embeddings, target, margin=1.0, squared=True):
        """
        :param features: [B * N features]
        :param target: [B]
        :param square: if the distance squared or not.
        :return:
        """
        #print('emb',embeddings)
        #print('target',target)
        #print(margin)
        lshape = target.shape
        assert len(lshape) == 1
        labels = target.int().unsqueeze(-1)  # [B, 1]
        pdist_matrix = self.pairwise_distance(embeddings, squared=squared)

        adjacency = labels == torch.transpose(labels, 0, 1)

        adjacency_not = ~adjacency
        batch_size = labels.shape[0]

        # Compute the mask

        pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

        mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(
            torch.transpose(pdist_matrix, 0, 1), [-1, 1]))

        mask_final = torch.reshape(torch.sum(mask.float(), 1, keepdim=True) >
                                   0.0, [batch_size, batch_size])
        mask_final = torch.transpose(mask_final, 0, 1)

        adjacency_not = adjacency_not.float()
        mask = mask.float()

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = torch.reshape(
            self.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = torch.transpose(negatives_outside, 0, 1)

        # negatives_inside: largest D_an.
        negatives_inside = self.masked_maximum(pdist_matrix, adjacency_not).repeat([1, batch_size])
        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = torch.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = adjacency.float() - torch.diag(cudafy(torch.ones([batch_size])))
        #mask_positives = adjacency.float() - torch.diag(torch.ones([batch_size]))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = torch.sum(mask_positives)

        triplet_loss = torch.div(torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0)), num_positives)

        # triplet_loss = torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0))
        return triplet_loss

#unsupervised
def train_unsup(args, model, device, dataset, optimizer):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        print('batch1.x',batch1.x)

        print('batch1.y', batch1.y)

        x1, _ = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.y, batch1.batch)
        x2, _ = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch1.y, batch2.batch)
        loss = model.loss_unsup(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)

#supervised:supcl loss,semehard loss
def train_supcl(args, model, device, dataset, optimizer):

    dataset = dataset.shuffle()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    model.train()
    optimizer.zero_grad()

    train_loss_accum = 0
    loss_list = []

    for step, batch in tqdm(enumerate(loader), desc='Iteration'):
        batch.to(device)
        x, y = model.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch)
        if args.mode == 'supcl':
            loss = model.loss_supcl(x, y)
        elif args.mode == 'semihard':
            loss = model.loss_semihard(x, y, 1)


        loss.backward()
        optimizer.step()

        print(loss)


        temp = loss.detach().cpu().item()
        print(temp)
        #train_loss_accum += float(loss.detach().cpu().item())
        train_loss_accum += float(temp)
        loss_list = loss_list + [temp]

    #return train_loss_accum
    return loss_list, train_loss_accum


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.01)')#defalut:0.001
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'cliff_custom', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = './models_graphcl/cliff_custom/addsetgraphcl_80.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='none')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default='none')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    parser.add_argument('--mode', type=str, default='semihard', help='choice of pretrain: unsup,supcl,semihard')
    parser.add_argument('--feature_type', type=str, default='custom', help='random, onehot, custom, psudo')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset
    dataset = MoleculeDataset_aug("dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)
    print(dataset.__dict__)
    #read smiles_list
    smiles_list = list(pd.read_csv('./dataset/cliff_custom/processed/smiles.csv',header = None)[0])
    #print(smiles_list)
    for i in smiles_list:
        visual_smiles.append(i)
    '''
    for i in smiles_list:
        visual_dict[i] = []
    print(visual_dict)
    return
    
    '''
    '''
    for 
    
    f.open('visual.txt','a')
    '''
    
    
    
    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio,
              feature_type = args.feature_type,
              gnn_type = args.gnn_type)
    
    
    
    
    model = graphcl(gnn)
    model.to(device)
    model.load_state_dict(torch.load(args.input_model_file))
    
    loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)
    
    
    for step, batch in tqdm(enumerate(loader), desc='Iteration'):
        batch.to(device)

        x, y = model.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch)
        print('Labels:',y,y.shape)
    
    assert len(visual_smiles)==len(visual_weight)
    
    
    visual_dict = dict(zip(visual_smiles,visual_weight))
    print(visual_dict['O=C(NCCC(C)(C)C)C(Cc1cc2cc(ccc2nc1N)-c1ncccc1C#N)C'],visual_dict['O=C(NCCC(C)(C)C)C(Cc1cc2cc(ccc2nc1N)-c1ncccc1C)C'])
    
    exp = []
    for i in range(5):
        exp.append(visual_dict[i])
    
    norm = matplotlib.colors.Normalize(vmin=0,vmax=1.28)
    cmap = cm.get_cmap('bwr')
    plt_colors = cm.ScalarMappable(norm = norm,cmap=cmap)
    for i in exp:
        minval = min(i[1])
        maxval = max(i[1])
        atom_weight = np.array(i[1])
        atom_weight = list((atom_weight-minval)/(maxval-minval))
        atom_colors = {j:plt_colors.to_rgba(i[1][j]) for j in range(len(i[1]))}
        
        mol = Chem.MolFromSmiles(i[0])
        
if __name__ == "__main__":
    main()
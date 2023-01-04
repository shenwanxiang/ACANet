# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 11:08:35 2022

@author: wanxiang.shen@u.nus.edu
"""


import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import numpy as np
import pandas as pd
from tqdm import tqdm

class ACALoss(_Loss):
    r"""Creates a criterion that measures the activity cliff awareness (ACA) loss given an input
    tensors :math:`y_true`, :math:`y_pred`, :math:`y_emb`, an awareness factor :math:`𝑎` 
    and two cliff parameters :math:`cliff_lower` and :math:`cliff_upper` with a value greater than :math:`0`.

    This is used for increasing the activty cliff awareness in regression tasks of molecular property prediction. 
    It consists of two parts, the natural regression loss of mae or mse and an active cliff perception loss in latent space. 
    The ACALoss is described in detail in the paper `Online triplet contrastive learning enables efficient cliff 
    awareness in regression tasks of molecular property prediction`.


    The loss function for each sample in the mini-batch is:

    .. math::
        L(aca) = L(mae/mse) + 𝑎 * L(tsm)
        L(tsm) = ∑_(j=1)^M[|f_j^a-f_j^p |-|f_j^a-f_j^n |+m_j ]_+ 
        
    where the L_mae is the MAE loss, the L_tsm is the triplet loss with soft margin, 𝑎 is the awareness factor, 
    N is the number of samples in each batch, M is the number of the mined (i.e., the valid) triplets in each batch, 
    the y_j, f_j are the j-th true label and latent vectors, respectively. 
    The item m_j is the soft margin of the mined j-th triplet and is defined by:
    
    .. math::
        m_j=|y_j^a-y_j^n |-|y_j^a-y_j^p |

    where `a`, `p` and `n` are `anchor`, `positive` and `negative` examples of a mined triplet in a min-batch, respectively

    It can be seen that the L_tsm term is only determined by the true labels and the embedding vectors in the latent space.
    Therefore, this term is forcing the model to learn active cliffs in the latent space.
    

    Args:
        alpha (float, optional): awareness factor. Default: :math:`1.0`.
        cliff_lower (float, optional): The threshold for mining the postive samples. Default: ``1.0``
        cliff_upper (float, optional): The threshold for mining the negative samples. Default: ``1.0``
        squared (bool, optional): if True, the mse loss will be used, otherwise mae. The L(tsm) will also be squared.
        dev_mode (bool, optional): if False, only return the union loss
    Examples::
    ## developer mode
    >>> aca_loss = ACALoss(alpha=1.0, cliff_lower = 0.2, cliff_upper = 1.0, squared = True, dev_mode = True)
    >>> loss, reg_loss, tsm_loss, n_mined_triplets, n_pos_triplets = aca_loss(labels, predictions, embeddings)
    >>> loss.backward()
    ## normal mode
    >>> aca_loss = ACALoss(dev_mode = False)
    >>> loss = aca_loss(labels, predictions, embeddings)
    >>> loss.backward()
    
    """
    
    __constants__ = ['alpha', 'cliff_lower', 'cliff_upper', 'squared']
    alpha: float
    cliff_lower: float
    cliff_upper: float
    squared: bool
    dev_mode: bool

    def __init__(self, 
                 alpha: float = 1.0, 
                 cliff_lower: float = 1.0, 
                 cliff_upper: float = 1.0,
                 squared: bool = False, 
                 dev_mode = True
                ):
        
        super(ACALoss, self).__init__(alpha)
        self.alpha = alpha
        self.cliff_lower = cliff_lower
        self.cliff_upper = cliff_upper
        self.squared = squared
        self.dev_mode = dev_mode
        
    def forward(self, labels: Tensor, predictions: Tensor, embeddings: Tensor) -> Tensor:

        return _aca_loss(labels, predictions, embeddings, alpha=self.alpha, 
                        cliff_lower=self.cliff_lower, cliff_upper=self.cliff_upper,
                        squared=self.squared, dev_mode = self.dev_mode)
    

def pairwise_distance(embeddings, squared=True):
    pdist = torch.cdist(embeddings, embeddings, p = 1)
    if squared:
        pdist = pdist**2
    return pdist


    
def get_triplet_mask(labels, device, cliff_lower = 0.2, cliff_upper = 1.0):

    indices_equal = torch.eye(labels.shape[0]).bool()
    indices_not_equal = torch.logical_not(indices_equal)
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = torch.logical_and(torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k).to(device)

    #labels = torch.unsqueeze(labels, -1)
    #print('labels:',labels)
    target_l1_dist = torch.cdist(labels,labels,p=1) 
    label_equal = target_l1_dist < cliff_lower #0.5
    label_unequal  = target_l1_dist >= cliff_upper #1.5
    
    #print('label_equal:',label_equal)
    i_equal_j = torch.unsqueeze(label_equal, 2)
    #i_equal_k = torch.unsqueeze(label_equal, 1)
    i_unequal_k = torch.unsqueeze(label_unequal, 1)
    
    valid_labels = torch.logical_and(i_equal_j, i_unequal_k).to(device)
    #print('val_indice',valid_labels[0])

    mask = torch.logical_and(distinct_indices, valid_labels)
    return mask   



def _aca_loss(labels,
              predictions, 
              embeddings,
              alpha=1.0,
              cliff_lower=0.2,
              cliff_upper=1.0,
              squared = False,
              dev_mode = True
              ):
    '''
       union loss of a batch (mae loss and triplet loss with soft margin)
       -------------------------------
       Args:
          labels: shape = （batch_size,）
          predictions: shape = （batch_size,） 
          embeddings: shape = (batch_size, embedding_vector_size)
          alpha (float, optional): awareness factor. Default: :math:`1.0`.
          cliff_lower (float, optional): The threshold for mining the postive samples. Default: ``1.0``
          cliff_upper (float, optional): The threshold for mining the negative samples. Default: ``1.0``
          squared (bool, optional): if True, the mse loss will be used, otherwise mae. The L(tsm) will also be squared.
       Returns:
         loss, reg_loss, tsm_loss, n_mined_triplets, n_pos_triplets
    '''

    if squared:
        reg_loss = torch.mean((labels-predictions).abs()**2)
    else:
        reg_loss = torch.mean((labels-predictions).abs())

    device = embeddings.device
    
    # label pairwise distance for soft margin
    labels_dist = pairwise_distance(embeddings=labels, squared=squared)
    margin_pos = labels_dist.unsqueeze(2)
    margin_neg = labels_dist.unsqueeze(1)
    margin = margin_neg - margin_pos

    # embedding pairwise distance for (a, p, n) mining
    pairwise_dis = pairwise_distance(embeddings=embeddings, squared=squared)
    anchor_positive_dist = pairwise_dis.unsqueeze(2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(
        anchor_positive_dist.shape)
    anchor_negative_dist = pairwise_dis.unsqueeze(1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(
        anchor_negative_dist.shape)
    #triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_mask(labels=labels, device=device,
                             cliff_lower=cliff_lower, cliff_upper=cliff_upper)
    mask = mask.float()
    n_mined_triplets = torch.sum(mask)  # total number of mined triplets
    
    triplet_loss = torch.mul(mask, triplet_loss)
    triplet_loss = torch.maximum(triplet_loss, torch.tensor([0.0]).to(device))

    # Count the number of the triplet that > 0. Should be decreased with increasing of the epochs
    pos_triplets = (triplet_loss > 1e-16).float()
    n_pos_triplets = torch.sum(pos_triplets)  # torch.where

        
    tsm_loss = torch.sum(triplet_loss) / (n_mined_triplets + 1e-16)
    
    loss = reg_loss + alpha*tsm_loss
    
    if dev_mode:
        return loss, reg_loss, tsm_loss, n_mined_triplets, n_pos_triplets
    else:
        return loss
    


def get_best_cliff(labels, cliffs = list(np.arange(0.1, 3.2, 0.1).round(2))):
    '''
    Get the best cliff lower and upper values. Under these value, we can mine the maximal No. of triplets.
    '''

    low_up_trps = []
    n = 0
    best_lower = 0
    best_upper = 0
    for lower in cliffs:
        for upper in cliffs:
            if upper >= lower:
                mask = get_triplet_mask(
                    labels, labels.device, cliff_lower=lower, cliff_upper=upper)
                mask = mask.float()
                n_mined_trps = int(torch.sum(mask).cpu().numpy())
                if n_mined_trps > n:
                    n = n_mined_trps
                    best_lower = lower
                    best_upper = upper

    return best_lower, best_upper, n


def get_best_cliff_batch(train_dataset, 
                         device,
                         batch_size=128, 
                         epochs = 10, 
                         cliffs = list(np.arange(0.1, 3.2, 0.1).round(2))):
    
    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print('Find the potential cliff parameters automatically...')
    c = []
    for epoch in tqdm(range(epochs), desc = 'epoch', ascii=True):
        for data in train_loader:
            cl, cu, _ = get_best_cliff(data.y.to(device), cliffs = cliffs)
            c.append([cl, cu, epoch])
    c_distribution = pd.DataFrame(c).groupby(0).size()/len(c)
    best_cliff = c_distribution.idxmax()
    return best_cliff



def get_best_cliff_exp(labels, cliffs = list(np.arange(0.1, 3.2, 0.1).round(2))):
    '''
    Get the best cliff lower and upper values. Under these value, we can mine the maximal triplets.
    '''
    
    low_up_trps = []
    for lower in cliffs:
        for upper in cliffs:
            if upper >= lower:
                mask = get_triplet_mask(
                    labels, labels.device, cliff_lower=lower, cliff_upper=upper)
                if upper == lower:
                    split = 1
                else:
                    split = 2
                mask = mask.float()
                n_mined_trps = torch.sum(mask).cpu().numpy()
                low_up_trps.append([lower, upper, int(n_mined_trps), split])
                
    df = pd.DataFrame(low_up_trps, columns = ['lower', 'upper', 'trps', 'split'])
    _best = df.groupby('split').apply(lambda x: x.sort_values('trps').iloc[-1:])
    s1_best = _best.loc[1].iloc[0].to_dict()
    s2_best = _best.loc[2].iloc[0].to_dict()
    
    return s1_best, s2_best, df
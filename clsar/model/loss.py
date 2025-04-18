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
from Levenshtein import distance as levenshtein_distance

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
        alpha (float, optional): awareness factor. Default: :math:`0.1`.
        cliff_lower (float, optional): The threshold for mining the postive samples. Default: ``1.0``
        cliff_upper (float, optional): The threshold for mining the negative samples. Default: ``1.0``
        squared (bool, optional): if True, the mse loss will be used, otherwise mae. The L(tsm) will also be squared.
        p (float, optional) – p value for the p-norm distance to calculate the distance of latent vectors ∈[0,∞]. Default: ``2.0``
        dev_mode (bool, optional): if False, only return the union loss
    Examples::
    ## developer mode
    >>> aca_loss = ACALoss(alpha=0.1, cliff_lower = 0.2, cliff_upper = 1.0, p = 1., squared = True, dev_mode = True)
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
    p: float
    dev_mode: bool

    def __init__(self, 
                 alpha: float = 0.1, 
                 cliff_lower: float = 1.0, 
                 cliff_upper: float = 1.0,
                 squared: bool = False, 
                 p: float = 2.0,
                 dev_mode = True,
                 **kwargs,
                ):
        
        super(ACALoss, self).__init__(alpha)
        self.alpha = alpha
        self.cliff_lower = cliff_lower
        self.cliff_upper = cliff_upper
        self.squared = squared
        self.p = p
        self.dev_mode = dev_mode

        self.fp_filter = kwargs.get('fp_filter', None)
        self.scaffold_filter = kwargs.get('scaffold_filter', None)
        self.smiles_filter = kwargs.get('smiles_filter', None)
        
    def forward(self, labels: Tensor, predictions: Tensor, embeddings: Tensor, fp_values, scaffold_values, smiles_values) -> Tensor:
        return _aca_loss(labels, predictions, embeddings, alpha=self.alpha, 
                        cliff_lower=self.cliff_lower, cliff_upper=self.cliff_upper,
                        squared=self.squared, p = self.p, dev_mode = self.dev_mode, fp_filter = self.fp_filter, scaffold_filter = self.scaffold_filter, smiles_filter= self.smiles_filter, fp_values = fp_values, scaffold_values = scaffold_values, smiles_values = smiles_values)
    

def pairwise_distance(embeddings, p = 2, squared=True):
    pdist = torch.cdist(embeddings, embeddings, p = p)
    
    ## normalized l1/l2 distance along the vector size
    # N = np.power(embeddings.shape[1], 1/p)
    # pdist = pdist / N
    
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
    target_l1_dist = torch.cdist(labels,labels, p=1) 
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

# Get abosolute struct mask which selects samples that satisfy FP similarity constrains. e.g. Tanimoto Similarity
def get_fp_mask(fps, device, struct_threshould_neg=0.9, struct_threshould_pos=1, eps=1e-5):
    
    assert len(fps.size())==2, 'The FP shape should be [batch, fingerprint_dim]'
    common = torch.bitwise_and(fps.unsqueeze(0),fps.unsqueeze(1))
    a_add_b = torch.add(fps.unsqueeze(0),fps.unsqueeze(1))
    
    C = common.sum(-1)
    A_add_B = a_add_b.sum(-1)
    similarity = C / (A_add_B- C + eps)
    sim_mask_neg = similarity > struct_threshould_neg
    sim_mask_pos = similarity < struct_threshould_pos
    # The same as former triplet mask, dim 'j' denotes positive sample, dim 'k' denotes negative samples
    mask = torch.logical_and(sim_mask_neg.unsqueeze(2), sim_mask_neg.unsqueeze(1)).to(device)
    return mask

# Get abosolute struct mask which selects samples that satisfy FP similarity constrains. e.g. Tanimoto Similarity
def get_scaffold_mask(scaffold_fps, device, struct_threshould_neg=0.9, struct_threshould_pos=1, eps=1e-5):
    fps = scaffold_fps
    assert len(fps.size())==2, 'The FP shape should be [batch, fingerprint_dim]'
    common = torch.bitwise_and(fps.unsqueeze(0),fps.unsqueeze(1))
    a_add_b = torch.add(fps.unsqueeze(0),fps.unsqueeze(1))
    
    C = common.sum(-1)
    A_add_B = a_add_b.sum(-1)
    similarity = C / (A_add_B- C + eps)
    sim_mask_neg = similarity > struct_threshould_neg
    sim_mask_pos = similarity < struct_threshould_pos
    # The same as former triplet mask, dim 'j' denotes positive sample, dim 'k' denotes negative samples
    mask = torch.logical_and(sim_mask_neg.unsqueeze(2), sim_mask_neg.unsqueeze(1)).to(device)
    return mask
# def build_scaffold_id_map(scaffold_list):
#     unique_scaffolds = set(scaffold_list)
#     scaffold2id = {scf: idx for idx, scf in enumerate(unique_scaffolds)}
#     return scaffold2id
# def get_scaffold_mask(scaffold_list, scaffold2id, device):
#     batch_ids = []
#     for scf in scaffold_list:
#         if scf in scaffold2id:
#             batch_ids.append(scaffold2id[scf])
#         else:
#             batch_ids.append(-1)
    
#     batch_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
#     B = batch_ids.size(0)

#     pos_mask = (batch_ids.unsqueeze(0) == batch_ids.unsqueeze(1))  # shape [B, B]
#     neg_mask = (batch_ids.unsqueeze(0) != batch_ids.unsqueeze(1))  # shape [B, B]

#     pos_mask_3d = pos_mask.unsqueeze(1)  # [B, B, 1]
#     neg_mask_3d = neg_mask.unsqueeze(2)  # [B, 1, B]
    
#     mask_3d = pos_mask_3d & neg_mask_3d  # [B, B, B]
    
#     return mask_3d

def compute_edit_similarity(s1, s2):
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - dist / max_len

def get_smiles_mask(smiles_list, 
                    device,
                    struct_threshold_neg=0.9, 
                    struct_threshold_pos=1.0, 
                    ):
    B = len(smiles_list)
    
    edit_sim = torch.zeros(B, B, dtype=torch.float)
    for i in range(B):
        for j in range(B):
            if i == j:
                edit_sim[i, j] = 1.0
            else:
                edit_sim[i, j] = compute_edit_similarity(smiles_list[i], smiles_list[j])
    
    neg_mask_2d = edit_sim >= struct_threshold_neg
    pos_mask_2d = edit_sim < struct_threshold_pos
    
    neg_mask_3d = neg_mask_2d.unsqueeze(1)  # shape: [B, 1, B]
    pos_mask_3d = pos_mask_2d.unsqueeze(2)  # shape: [B, B, 1]
    
    mask_3d = torch.logical_and(neg_mask_3d, pos_mask_3d).to(device)
    return mask_3d

def _aca_loss(labels,
              predictions, 
              embeddings,
              alpha=0.1,
              cliff_lower=0.2,
              cliff_upper=1.0,
              squared = False,
              p = 2.0,
              dev_mode = True,
              **kwargs
              ):
    '''
       union loss of a batch (mae loss and triplet loss with soft margin)
       -------------------------------
       Args:
          labels: shape = （batch_size,）
          predictions: shape = （batch_size,） 
          embeddings: shape = (batch_size, embedding_vector_size)
          alpha (float, optional): awareness factor. Default: :math:`0.1`.
          cliff_lower (float, optional): The threshold for mining the postive samples. Default: ``1.0``
          cliff_upper (float, optional): The threshold for mining the negative samples. Default: ``1.0``
          p (float, optional) – p value for the p-norm distance to calculate between each vector pair ∈[0,∞].
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
    labels_dist = pairwise_distance(embeddings=labels, p = p, squared=squared)
    margin_pos = labels_dist.unsqueeze(2)
    margin_neg = labels_dist.unsqueeze(1)
    margin = margin_neg - margin_pos
    #margin = torch.maximum(margin, torch.tensor([0.0]).to(device))
    
    # embedding pairwise distance for (a, p, n) mining
    pairwise_dis = pairwise_distance(embeddings=embeddings, p = p, squared=squared)
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
    full_mask = mask.clone()
    filter_mask = torch.zeros_like(mask)
    if kwargs.get('fp_filter', None) and kwargs.get('fp_values', None) is not None:
        fps = kwargs.get('fp_values')
        fp_mask = get_fp_mask(fps=fps, device=device)
        filter_mask = torch.bitwise_or(filter_mask, fp_mask)
    if kwargs.get('scaffold_filter', None) and kwargs.get('scaffold_values', None) is not None:
        scaffold_fps = kwargs.get('scaffold_values')
        # scaffold2id = build_scaffold_id_map(scaffold_list)
        scaffold_mask = get_scaffold_mask(scaffold_fps=scaffold_fps, device=device)
        filter_mask = torch.bitwise_or(filter_mask, scaffold_mask)
    if kwargs.get('smiles_filter', None) and kwargs.get('smiles_values', None) is not None:
        smiles_list = kwargs.get('smiles_values')
        smiles_mask = get_smiles_mask(smiles_list, device=device)
        filter_mask = torch.bitwise_or(filter_mask, smiles_mask)
    full_mask = full_mask*filter_mask

    n_mined_triplets_origin, n_pos_triplets_origin = -1, -1
    if kwargs.get('fp_filter', None) or kwargs.get('scaffold_filter', None) or kwargs.get('smiles_filter', None):
        mask = mask.float()
        n_mined_triplets_origin = torch.sum(mask)  # origin number of mined triplets
        triplet_loss_origin = torch.mul(mask, triplet_loss)
        triplet_loss_origin = torch.maximum(triplet_loss_origin, torch.tensor([0.0]).to(device))
        pos_triplets_origin = (triplet_loss_origin > 1e-16).float()
        n_pos_triplets_origin = torch.sum(pos_triplets_origin)

    # Count the number of the triplet that > 0. Should be decreased with increasing of the epochs
    pos_triplets = (triplet_loss > 1e-16).float()
    n_pos_triplets = torch.sum(pos_triplets)  # torch.where

    full_mask = full_mask.float()
    n_mined_triplets = torch.sum(full_mask)  # origin number of mined triplets

    triplet_loss = torch.mul(full_mask, triplet_loss)
    triplet_loss = torch.maximum(triplet_loss, torch.tensor([0.0]).to(device))

    # Count the number of the triplet that > 0. Should be decreased with increasing of the epochs
    pos_triplets = (triplet_loss > 1e-16).float()
    n_pos_triplets = torch.sum(pos_triplets)  # torch.where

    # tsm_loss = torch.sum(triplet_loss) / (n_mined_triplets + 1e-16)
    # loss = reg_loss + alpha*tsm_loss

    if n_mined_triplets == 0:
        tsm_loss = n_mined_triplets.float()
        loss = reg_loss 

    else:
        tsm_loss = torch.sum(triplet_loss) / n_mined_triplets
        loss = reg_loss + alpha*tsm_loss
        
    if dev_mode:
        return loss, reg_loss, tsm_loss, n_mined_triplets, n_pos_triplets , n_mined_triplets_origin, n_pos_triplets_origin
    else:
        return loss
    


#Add absolute structure mask of Tanimoto Similarity
def _fp_aca_loss(labels,
              predictions, 
              embeddings,
              fps,
              alpha=0.1,
              cliff_lower=0.2,
              cliff_upper=1.0,
              sim_threshould_neg=0.9,
              sim_threshould_pos=1,
              squared = False,
              p = 2.0,
              dev_mode = True,
              **kwargs
              ):
    '''
       union loss of a batch (mae loss and triplet loss with soft margin)
       -------------------------------
       Args:
          labels: shape = （batch_size,）
          predictions: shape = （batch_size,） 
          embeddings: shape = (batch_size, embedding_vector_size)
          alpha (float, optional): awareness factor. Default: :math:`0.1`.
          cliff_lower (float, optional): The threshold for mining the postive samples. Default: ``1.0``
          cliff_upper (float, optional): The threshold for mining the negative samples. Default: ``1.0``
          sim_threshould (float,optional): The threshold for mining the samples with simlilar struncture . Default: ``0.9``
          p (float, optional) – p value for the p-norm distance to calculate between each vector pair ∈[0,∞].
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
    labels_dist = pairwise_distance(embeddings=labels, p = p, squared=squared)
    margin_pos = labels_dist.unsqueeze(2)
    margin_neg = labels_dist.unsqueeze(1)
    margin = margin_neg - margin_pos
    #margin = torch.maximum(margin, torch.tensor([0.0]).to(device))
    
    # embedding pairwise distance for (a, p, n) mining
    pairwise_dis = pairwise_distance(embeddings=embeddings, p = p, squared=squared)
    anchor_positive_dist = pairwise_dis.unsqueeze(2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(
        anchor_positive_dist.shape)
    anchor_negative_dist = pairwise_dis.unsqueeze(1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(
        anchor_negative_dist.shape)
    #triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = get_triplet_fp_mask(labels=labels, fps=fps, device=device,
                             cliff_lower=cliff_lower, cliff_upper=cliff_upper,
                             sim_threshould_neg=sim_threshould_neg, sim_threshould_pos=sim_threshould_pos)
    mask = mask.float()
    n_mined_triplets = torch.sum(mask)  # total number of mined triplets
    
    triplet_loss = torch.mul(mask, triplet_loss)
    triplet_loss = torch.maximum(triplet_loss, torch.tensor([0.0]).to(device))

    # Count the number of the triplet that > 0. Should be decreased with increasing of the epochs
    pos_triplets = (triplet_loss > 1e-16).float()
    n_pos_triplets = torch.sum(pos_triplets)  # torch.where

    # tsm_loss = torch.sum(triplet_loss) / (n_mined_triplets + 1e-16)
    # loss = reg_loss + alpha*tsm_loss

    if n_mined_triplets == 0:
        tsm_loss = n_mined_triplets.float()
        loss = reg_loss 

    else:
        tsm_loss = torch.sum(triplet_loss) / n_mined_triplets
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
                low_up_trps.append([lower, upper, int(n_mined_trps)])
                
                if n_mined_trps > n:
                    n = n_mined_trps
                    best_lower = lower
                    best_upper = upper
    
    df = pd.DataFrame(low_up_trps, columns = ['lower', 'upper', 'trps'])
    
    return best_lower, best_upper, df


def get_best_cliff_batch(train_dataset, 
                         device,
                         batch_size=128, 
                         iterations = 10, 
                         cliffs = list(np.arange(0.1, 3.2, 0.1).round(2))):
    
    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print('Find the potential cliff parameters automatically...')
    c = []
    for epoch in tqdm(range(iterations), desc = 'epoch', ascii=True):
        for data in train_loader:
            cl, cu, _ = get_best_cliff(data.y.to(device), cliffs = cliffs)
            c.append([cl, cu, epoch])
    c_distribution = pd.DataFrame(c).groupby(0).size()/len(c)
    best_cliff = c_distribution.idxmax()
    return best_cliff


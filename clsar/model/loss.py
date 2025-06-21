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
from torch_geometric.loader import DataLoader
from typing import Optional, List
from rdkit import DataStructs

class ACALoss(_Loss):
    r"""Criterion for Activity Cliff Awareness (ACA) loss.
    Combines a regression loss (MAE or MSE) with a triplet-soft-margin loss
    to emphasize activity cliffs in latent space.

    Args:
        alpha: weight of the triplet loss term.
        cliff_lower: threshold for hard-positive in label space.
        cliff_upper: threshold for hard-negative in label space.
        squared: if True, use MSE; else MAE.
        p: p-norm for embedding distance.
        similarity_gate: if True, filter triplets by structural similarity.
        similarity_neg: Tanimoto threshold for positives.
        similarity_pos: Tanimoto threshold for negatives.
        gate_type: 'AND' or 'OR' combining label and structure masks.
        dev_mode: if True, return detailed stats.
    """

    __constants__ = ['alpha', 'cliff_lower', 'cliff_upper', 'squared',
                     'p', 'similarity_gate', 'similarity_neg', 'similarity_pos', 'dev_mode', 'gate_type']

    def __init__(self,
                 alpha: float = 0.1,
                 cliff_lower: float = 1.0,
                 cliff_upper: float = 1.0,
                 squared: bool = False,
                 p: float = 2.0,
                 similarity_gate: bool = False,
                 similarity_neg: float = 0.9,
                 similarity_pos: float = 0.1,
                 gate_type: str = 'AND',
                 dev_mode: bool = False):
        super(ACALoss, self).__init__(alpha)
        self.alpha = alpha
        self.cliff_lower = cliff_lower
        self.cliff_upper = cliff_upper
        self.squared = squared
        self.p = p
        self.similarity_gate = similarity_gate
        self.similarity_neg = similarity_neg
        self.similarity_pos = similarity_pos
        self.gate_type = gate_type.upper()
        self.dev_mode = dev_mode

    def forward(self,
                labels: Tensor,
                predictions: Tensor,
                embeddings: Tensor,
                fps_smiles: Optional[Tensor] = None,
                fps_scaffold: Optional[Tensor] = None,
                smiles_list: Optional[List[str]] = None) -> Tensor:
        
        return _aca_loss(
            labels=labels,
            predictions=predictions,
            embeddings=embeddings,

            # structure
            fps_smiles=fps_smiles,
            fps_scaffold=fps_scaffold,
            smiles_list=smiles_list,

            # parameters
            alpha=self.alpha,
            cliff_lower=self.cliff_lower,
            cliff_upper=self.cliff_upper,
            similarity_gate=self.similarity_gate,
            similarity_neg=self.similarity_neg,
            similarity_pos=self.similarity_pos,
            gate_type=self.gate_type,
            squared=self.squared,
            p=self.p,
            dev_mode=self.dev_mode
        )

    
def pairwise_distance(embeddings: torch.Tensor, p: float = 2, squared: bool = True) -> torch.Tensor:
    """
    计算 embeddings 间的成对距离：
    - embeddings: [B, D] 张量
    - p: Lp 范数
    - squared: 如果 True，则返回平方距离；否则返回原始距离
    """
    pdist = torch.cdist(embeddings, embeddings, p=p)  # [B, B]
    if squared:
        pdist = pdist ** 2
    return pdist  # [B, B]


def get_label_mask(labels: torch.Tensor,
                   device: torch.device,
                   cliff_lower: float = 0.2,
                   cliff_upper: float = 1.0) -> torch.Tensor:
    """
    构造基于标签（活性）差的三元组掩码，确保正负条件为：
      - “hard positive” (i,j)： |labels[i] - labels[j]| < cliff_lower
      - “hard negative” (i,k)： |labels[i] - labels[k]| ≥ cliff_upper
    并且 i, j, k 三个索引两两不相等。

    Args:
        labels: 形状可以是 [B] 或 [B,1] 等任何形式，
                但内部会先 flatten 到 [B]。
        device: torch.device，用于将生成的掩码移动到同一设备。
        cliff_lower: float, “hard positive” 的最大 |Δy|。
        cliff_upper: float, “hard negative” 的最小 |Δy|。

    Returns:
        mask: 布尔张量，形状 [B, B, B]，表示 (i,j,k) 是否为有效三元组。
    """
    # 1. 将 labels 整成一维 [B]
    labels_flat = labels.view(-1)           # 如果原本是 [B,1]，这会变成 [B]
    B = labels_flat.size(0)

    # 2. 构造 “i≠j≠k” 的索引掩码
    idx = torch.arange(B, device=device)
    eq = idx.unsqueeze(0) == idx.unsqueeze(1)   # [B, B]，True 表示同索引
    neq = ~eq                                   # [B, B]，True 表示不同索引
    i_ne_j = neq.unsqueeze(2)                   # [B, B, 1]
    i_ne_k = neq.unsqueeze(1)                   # [B, 1, B]
    j_ne_k = neq.unsqueeze(0)                   # [1, B, B]
    distinct_indices = i_ne_j & i_ne_k & j_ne_k  # [B, B, B]，三者两两都不同

    # 3. 计算 labels_flat 之间的绝对距离矩阵
    #    labels_flat.unsqueeze(1): [B, 1]
    #    labels_flat.unsqueeze(0): [1, B]
    #    两者相减然后 abs → 得到 [B, B] 的 |y_i - y_j|
    labels_dist = torch.abs(labels_flat.unsqueeze(1) - labels_flat.unsqueeze(0))  # [B, B]

    # 4. 构造“hard positive” 与 “hard negative” 的 2D 掩码
    #    hard_pos[i,j] = True 当且仅当 |y_i - y_j| < cliff_lower
    #    hard_neg[i,k] = True 当且仅当 |y_i - y_k| ≥ cliff_upper
    hard_pos = labels_dist < cliff_lower    # [B, B]
    hard_neg = labels_dist >= cliff_upper   # [B, B]

    # 5. 扩展到三元组：dim=1 对应 j（正样本），dim=2 对应 k（负样本）
    i_eq_j = hard_pos.unsqueeze(2)   # [B, B, 1]
    i_uneq_k = hard_neg.unsqueeze(1) # [B, 1, B]
    valid_labels = i_eq_j & i_uneq_k  # [B, B, B]，同时满足 positive & negative 条件

    # 6. 最终掩码 = “索引都不同” & “标签条件满足”
    mask = distinct_indices & valid_labels  # [B, B, B]，布尔张量

    return mask


def compute_edit_similarity(s1, s2):
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - dist / max_len


def get_smiles_mask(smiles_list, 
                    device,
                    struct_threshold_neg=0.9, 
                    struct_threshold_pos=0.1, 
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


def get_fingerprint_mask(fingerprints: torch.Tensor,
                       device: torch.device,
                       struct_threshold_neg: float = 0.9,
                       struct_threshold_pos: float = 0.1,
                       eps: float = 1e-5) -> torch.Tensor:
    """
    构造基于指纹（fingerprint）相似度的三元组掩码：
    - fingerprints: BoolTensor，形状 [B, F] torch.tensor unit 8
    - struct_threshold_neg: Tanimoto 阈值，要求 sim(a, j) >= similarity_neg 才算“HARD 负样本候选”
    - struct_threshold_pos: Tanimoto 阈值，要求 sim(a, k) < similarity_pos 才算“HARD 正样本候选”
    返回：布尔型张量 mask，形状 [B, B, B]
    """


    fps = fingerprints.to(device)       # dtype=torch.uint8, shape [B, F]
    B = fps.size(0)
    
    # 1. 交集 ∧ 并集 ∨
    intersect = (fps.unsqueeze(1) & fps.unsqueeze(0)).sum(-1).float()  # [B, B]
    union     = (fps.unsqueeze(1) | fps.unsqueeze(0)).sum(-1).float()  # [B, B]
    
    # 2. Tanimoto
    sim_matrix = intersect / (union + 1e-8)   

    # 3. 构造 “正样本候选” 与 “负样本候选” 的 2D 掩码
    sim_hard_pos = sim_matrix < struct_threshold_pos  #0.1 [B, B]: i→j Tanimoto < threshold → 可做“硬正”
    sim_hard_neg = sim_matrix >= struct_threshold_neg #0.9 # [B, B]: i→k Tanimoto >= threshold → 可做“硬负”

    # 4. 扩展到 3D：dim=1 对应 j，dim=2 对应 k
    pos_3d = sim_hard_pos.unsqueeze(2)  # [B, B, 1]
    neg_3d = sim_hard_neg.unsqueeze(1)  # [B, 1, B]
    mask_3d = pos_3d & neg_3d              # [B, B, B]

    return mask_3d  # 布尔型张量


def get_structure_mask(fps_smiles: torch.Tensor,
                       fps_scaffold:torch.Tensor,
                       smiles_list,
                       device: torch.device,
                       similarity_neg: float = 0.9,
                       similarity_pos: float = 0.1,
                       eps: float = 1e-5) -> torch.Tensor:
    """
    构造基于指纹（fingerprint）相似度的三元组掩码：
    - fps_smiles, fps_scaffold: BoolTensor，形状 [B, F] torch.tensor unit 8
    - similarity_neg: Tanimoto 阈值，要求 Structure_sim(a, j) >= similarity_neg 才算“负样本候选” （要求负样本候选和anchor很相似）
    - similarity_pos: Tanimoto 阈值，要求 Structure_sim(a, k) < similarity_pos 才算“正样本候选”（要求正样本候选和anchor很不相似）
    返回：布尔型张量 mask，形状 [B, B, B]
    """
    
    fingerprint_mask = get_fingerprint_mask(fingerprints=fps_smiles, device=device, 
                                      struct_threshold_neg=similarity_neg, 
                                      struct_threshold_pos=similarity_pos)
    
    scaffold_mask = get_fingerprint_mask(fingerprints=fps_scaffold, device=device, 
                                      struct_threshold_neg=similarity_neg, 
                                      struct_threshold_pos=similarity_pos)
    
    smiles_mask = get_smiles_mask(smiles_list=smiles_list, device=device)



    # 再加上“索引两两不相等”的约束，保证 i, j, k 三者不同
    B = fps_smiles.size(0)
    idx = torch.arange(B, device=device)
    eq = idx.unsqueeze(0) == idx.unsqueeze(1)  # [B, B]
    neq = ~eq                                  # [B, B]
    i_ne_j = neq.unsqueeze(2)                # [B, B, 1]
    i_ne_k = neq.unsqueeze(1)                # [B, 1, B]
    j_ne_k = neq.unsqueeze(0)                # [1, B, B]
    distinct_idx = i_ne_j & i_ne_k & j_ne_k   # [B, B, B]
    
    final_mask = (fingerprint_mask | scaffold_mask | smiles_mask) & distinct_idx     # [B, B, B]
    
    return final_mask  # 布尔型张量




def _aca_loss(labels: torch.Tensor,
              predictions: torch.Tensor,
              embeddings: torch.Tensor,
              
              fps_smiles: torch.Tensor,
              fps_scaffold: torch.Tensor,
              smiles_list,
              
              alpha: float = 0.1,
              cliff_lower: float = 0.2,
              cliff_upper: float = 1.0,
              similarity_gate: bool = False,
              similarity_neg: float = 0.9,
              similarity_pos: float = 0.1,
              gate_type = 'AND',
              squared: bool = False,
              p: float = 2.0,
              dev_mode: bool = True,
              **kwargs):
    """
    Compute ACA loss = regression loss + alpha * triplet‐soft‐margin loss.

    Arguments:
        labels: [B] or [B,1] tensor of true values.
        predictions: [B] or [B,1] tensor of predicted values.
        embeddings: [B, E] tensor of latent vectors.
        
        fps_smiles, fps_scaffold: tensor of length B .
        
        alpha: weight for the triplet loss term.
        cliff_lower: threshold below which (|Δy| < cliff_lower) defines hard positives.
        cliff_upper: threshold at or above which (|Δy| ≥ cliff_upper) defines hard negatives.
        similarity_gate: if True, apply structural‐similarity filtering.
        similarity_neg: Tanimoto threshold for positives/negatives (sim >= similarity_neg).
        similarity_pos: Tanimoto threshold for negatives (sim < similarity_pos).
        gate_type: 
        squared: if True, use squared distances for both MSE and triplet; else use MAE.
        p: p‐norm for torch.cdist.
        dev_mode: if True, return extra statistics.

    Returns:
        If dev_mode=True:
            (loss, reg_loss, tsm_loss, N_Y_ACTs, N_S_ACTs, N_ACTs, N_HV_ACTs)
        Else:
            loss
    """
    device = embeddings.device
    # Flatten labels to shape [B]
    labels_flat = labels.view(-1)
    B = labels_flat.size(0)

    # 1. Regression loss (MAE or MSE)
    if squared:
        reg_loss = torch.mean((labels_flat - predictions.view(-1)).abs() ** 2)
    else:
        reg_loss = torch.mean((labels_flat - predictions.view(-1)).abs())

    # 2. Compute pairwise label distances for margin
    #    labels_flat.unsqueeze(1): [B, 1], labels_flat.unsqueeze(0): [1, B]
    labels_dist = torch.abs(labels_flat.unsqueeze(1) - labels_flat.unsqueeze(0))  # [B, B]
    margin_pos = labels_dist.unsqueeze(2)  # [B, B, 1]
    margin_neg = labels_dist.unsqueeze(1)  # [B, 1, B]
    margin = margin_neg - margin_pos       # [B, B, B]

    # 3. Compute pairwise embedding distances
    pairwise_dis = torch.cdist(embeddings, embeddings, p=p)  # [B, B]
    if squared:
        pairwise_dis = pairwise_dis ** 2
    anchor_pos_dist = pairwise_dis.unsqueeze(2)  # [B, B, 1]
    anchor_neg_dist = pairwise_dis.unsqueeze(1)  # [B, 1, B]
    triplet_loss_matrix = anchor_pos_dist - anchor_neg_dist + margin  # [B, B, B]

    # 4. Build “label‐based” triplet mask (bool)
    #    Condition: i≠j≠k, |y_i - y_j| < cliff_lower (positive), |y_i - y_k| ≥ cliff_upper (negative)
    idx = torch.arange(B, device=device)
    eq = idx.unsqueeze(0) == idx.unsqueeze(1)  # [B, B]
    neq = ~eq                                  # [B, B]
    i_ne_j = neq.unsqueeze(2)                  # [B, B, 1]
    i_ne_k = neq.unsqueeze(1)                  # [B, 1, B]
    j_ne_k = neq.unsqueeze(0)                  # [1, B, B]
    distinct = i_ne_j & i_ne_k & j_ne_k        # [B, B, B]

    hard_pos = labels_dist < cliff_lower        # [B, B]
    hard_neg = labels_dist >= cliff_upper       # [B, B]
    pos_3d = hard_pos.unsqueeze(2)              # [B, B, 1]
    neg_3d = hard_neg.unsqueeze(1)              # [B, 1, B]
    mask_by_y = distinct & (pos_3d & neg_3d)   # [B, B, B], bool
    N_Y_ACTs = int(mask_by_y.sum().item())

    # 5. Build “structure‐based” mask if requested; else all True
    if similarity_gate:
        mask_by_s = get_structure_mask(
            fps_smiles=fps_smiles,
            fps_scaffold=fps_scaffold,
            smiles_list=smiles_list,
            device=device,
            similarity_neg=similarity_neg,
            similarity_pos=similarity_pos
        )  # [B, B, B], bool

        N_S_ACTs = int(mask_by_s.sum().item())
        # 6. Combine label and structure masks

        if gate_type == 'AND':
            mask_full_bool = mask_by_y & mask_by_s
        else:
            mask_full_bool = mask_by_y | mask_by_s  # [B, B, B], bool
        N_ACTs = int(mask_full_bool.sum().item())
        
    else:
        mask_by_s = torch.ones_like(mask_by_y, dtype=torch.bool)
        N_S_ACTs = int(mask_by_s.sum().item())
        # 6. Combine label and structure masks
        mask_full_bool = mask_by_y
        N_ACTs = int(mask_full_bool.sum().item())

    # 7. Compute TSM loss on masked triplets
    if N_ACTs == 0:
        tsm_loss = torch.tensor(0.0, device=device)
        loss = reg_loss
        N_HV_ACTs = 0
    else:
        # Zero out entries not in mask_full_bool
        triplet_loss_masked = torch.zeros_like(triplet_loss_matrix)
        triplet_loss_masked[mask_full_bool] = triplet_loss_matrix[mask_full_bool]
        # Clamp negatives to zero
        triplet_loss_masked = torch.clamp(triplet_loss_masked, min=0.0)

        tsm_loss = triplet_loss_masked.sum() / N_ACTs
        loss = reg_loss + alpha * tsm_loss

        # Count how many masked triplets have positive loss
        pos_triplets_bool = triplet_loss_masked > 1e-16  # [B, B, B], bool
        N_HV_ACTs = int((pos_triplets_bool & mask_full_bool).sum().item())

    # 8. Return results
    if dev_mode:
        return loss, reg_loss, tsm_loss, N_Y_ACTs, N_S_ACTs, N_ACTs, N_HV_ACTs
    else:
        return loss


# N_Y_ACTs = mask_by_y.sum()
# N_S_ACTs = mask_by_s.sum()
# N_ACTs = (mask_by_y & mask_by_s).sum()
# N_HV_ACTs = ((triplet_loss_masked > 0) & mask_full_bool).sum()


    

def get_best_cliff(labels: torch.Tensor,
                   cliffs: list = list(torch.arange(0.1, 3.2, 0.1).tolist())):
    """
    Get the best (cliff_lower, cliff_upper) that maximizes the number of triplets
    mined purely based on y‐label (活性) 条件。

    Args:
        labels: [B] 张量，表示本批次的真实活性值。
        cliffs: 候选的 cliff_lower 和 cliff_upper 列表。

    Returns:
        best_lower: float, 最佳的 cliff_lower
        best_upper: float, 最佳的 cliff_upper
        df: DataFrame 包含每对 (lower, upper, n_triplets) 的列表
    """
    device = labels.device
    low_up_trps = []
    n_max = 0
    best_lower = 0.0
    best_upper = 0.0

    for lower in cliffs:
        for upper in cliffs:
            if upper < lower:
                continue
            mask = get_label_mask(
                labels, device,
                cliff_lower=lower, 
                cliff_upper=upper
            ).float()
            n_mined = int(mask.sum().item())
            low_up_trps.append([lower, upper, n_mined])

            if n_mined > n_max:
                n_max = n_mined
                best_lower = lower
                best_upper = upper

    df = pd.DataFrame(low_up_trps, columns=['lower', 'upper', 'trps'])
    return best_lower, best_upper, df


def get_best_cliff_batch(train_dataset,
                         device: torch.device,
                         batch_size: int = 128,
                         iterations: int = 10,
                         cliffs: list = list(torch.arange(0.1, 3.2, 0.1).tolist())):
    """
    在多批数据上反复调用 get_best_cliff，统计最常出现的最佳 (cliff_lower, cliff_upper)。

    Args:
        train_dataset: PyG-style Dataset 对象，Data.y 为活性。
        device: torch.device
        batch_size: int
        iterations: int，总共取多少个 epoch 的 batch 来统计
        cliffs: 候选的 cliff 参数列表

    Returns:
        best_lower: float，出现频率最高的 cliff_lower
        best_upper: float，出现频率最高的对应 cliff_upper
    """
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    records = []

    print("Finding best (cliff_lower, cliff_upper) across batches...")
    for epoch in tqdm(range(iterations), desc="Epochs", ascii=True):
        for data in loader:
            # 假设 data.y 是 [batch_size] 的活性张量
            y = data.y.to(device)
            lower, upper, _ = get_best_cliff(y, cliffs=cliffs)
            records.append((lower, upper))

    df_pairs = pd.DataFrame(records, columns=['lower', 'upper'])
    # 先找出现次数最多的 lower
    lower_mode = df_pairs['lower'].mode().iloc[0]
    # 在 lower = lower_mode 的行里，找出现次数最多的 upper
    upper_mode = df_pairs[df_pairs['lower'] == lower_mode]['upper'].mode().iloc[0]

    return lower_mode, upper_mode


get_best_label_threshold = get_best_cliff
get_best_label_batch = get_best_cliff_batch

def get_best_structure_threshold(fps_smiles,
                                 fps_scaffold,
                                 smiles_list,
                       neg_thresholds: list = list(torch.arange(0.5, 1.0, 0.05).tolist()),
                       pos_thresholds: list = list(torch.arange(0.0, 0.5, 0.05).tolist()),
                       device: torch.device = torch.device('cpu')):
    """
    Get the best (similarity_neg, similarity_pos) that maximizes the number of triplets
    mined purely based on结构相似度条件。

    Args:
        fps_smiles, fps_scaffold: 长度 B 的 tensor。
        neg_thresholds: Tanimoto 阈值列表，用于“硬正/硬负” (sim >= neg_threshold)。
        pos_thresholds: Tanimoto 阈值列表，用于“硬负” (sim < pos_threshold)。
        device: torch.device

    Returns:
        best_neg: float, 最佳的 similarity_neg
        best_pos: float, 最佳的 similarity_pos
        df: DataFrame 包含每对 (neg_threshold, pos_threshold, n_triplets) 的列表
    """
    B = len(fps_smiles)
    best_neg = 0.0
    best_pos = 0.0
    max_triples = -1
    records = []

    for neg in neg_thresholds:
        for pos in pos_thresholds:
            if pos >= neg:
                # 要求 pos < neg，否则没意义
                continue

            mask = get_structure_mask(
                fps_smiles=fps_smiles,
                fps_scaffold=fps_scaffold,
                smiles_list=smiles_list,
                device=device,
                similarity_neg=neg,
                similarity_pos=pos
            )  # [B, B, B]
            n_triples = int(mask.float().sum().item())
            records.append([neg, pos, n_triples])

            if n_triples > max_triples:
                max_triples = n_triples
                best_neg = neg
                best_pos = pos

    df = pd.DataFrame(records, columns=['neg_threshold', 'pos_threshold', 'n_triplets'])
    return best_neg, best_pos, df


def get_best_structure_batch(train_dataset,
                             device: torch.device,
                             batch_size: int = 128,
                             iterations: int = 10,
                             neg_thresholds: list = list(torch.arange(0.5, 1.0, 0.05).tolist()),
                             pos_thresholds: list = list(torch.arange(0.0, 0.5, 0.05).tolist())):
    """
    Across multiple batches, find the most frequently optimal
    (similarity_neg, similarity_pos) for structural triplet mining.

    Args:
        train_dataset: PyG-style Dataset，对象的 Data.fp 是长度为 batch_size 的指纹列表。
        device: torch.device
        batch_size: int
        iterations: int
        neg_thresholds: 候选的 similarity_neg 列表
        pos_thresholds: 候选的 similarity_pos 列表

    Returns:
        best_neg: float，出现最频繁的 neg_threshold
        best_pos: float，出现最频繁的对应 pos_threshold
    """
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    best_pairs = []

    print("Finding best (similarity_neg, similarity_pos) across batches...")
    for epoch in tqdm(range(iterations), desc="Epochs", ascii=True):
        for data in loader:
            # 假设 data.fp 是一个长度为 batch_size 的指纹列表
            fps_smiles = data.fps_smiles
            fps_scaffold = data.fps_scaffold
            smiles_list = data.smiles_list
            neg, pos, _ = get_best_structure_threshold(
                fps_smiles=fps_smiles,
                fps_scaffold = fps_scaffold,
                smiles_list = smiles_list,
                neg_thresholds=neg_thresholds,
                pos_thresholds=pos_thresholds,
                device=device
            )
            best_pairs.append((neg, pos))

    df_pairs = pd.DataFrame(best_pairs, columns=['neg', 'pos'])
    neg_mode = df_pairs['neg'].mode().iloc[0]
    pos_mode = df_pairs[df_pairs['neg'] == neg_mode]['pos'].mode().iloc[0]

    return neg_mode, pos_mode, df_pairs
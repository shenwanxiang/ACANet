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
# from Levenshtein import distance as levenshtein_distance

import torch
from torch import Tensor
from torch.nn import _Loss
from rdkit import DataStructs


class ACALoss(_Loss):
    r"""Creates a criterion that measures the activity cliff awareness (ACA) loss given input
    tensors :math:`labels`, :math:`predictions`, :math:`embeddings` and optional fingerprints,
    an awareness factor :math:`\alpha` and two cliff parameters :math:`cliff_lower` and :math:`cliff_upper`.

    This loss combines a standard regression loss (MAE or MSE) with a triplet‐soft‐margin (TSM) loss
    to emphasize “activity cliffs” in latent space.  See the docstring for `_aca_loss` for details.

    Args:
        alpha (float, optional): weight of the TSM term. Default: 0.1.
        cliff_lower (float, optional): minimum |Δy| for a “hard positive” pair. Default: 1.0.
        cliff_upper (float, optional): maximum |Δy| for a “hard positive” (and threshold for “hard negative”). Default: 1.0.
        squared (bool, optional): if True, use MSE for regression loss; otherwise MAE. Also squares the distances in TSM. Default: False.
        p (float, optional): p‐norm to use in torch.cdist for computing distances. Default: 2.0.
        similarity_gate (bool, optional): if True, require structural‐similarity gating (Tanimoto thresholds). Default: False.
        similarity_neg (float, optional): Tanimoto threshold for “hard positive”/“hard negative” gating (sim > similarity_neg). Default: 0.8.
        similarity_pos (float, optional): Tanimoto threshold below which a pair is considered “dissimilar” (sim < similarity_pos) for negative sampling. Default: 0.2.
        dev_mode (bool, optional): if True, forward returns extra statistics: (loss, reg_loss, tsm_loss, N_Y_ACTs, N_S_ACTs, N_ACTs, N_HV_ACTs). If False, returns only the total loss. Default: True.
    """

    __constants__ = ['alpha', 'cliff_lower', 'cliff_upper', 'squared',
                     'p', 'similarity_gate', 'similarity_neg', 'similarity_pos', 'dev_mode']
    alpha: float
    cliff_lower: float
    cliff_upper: float
    squared: bool
    p: float
    similarity_gate: bool
    similarity_neg: float
    similarity_pos: float
    dev_mode: bool

    def __init__(self,
                 alpha: float = 0.1,
                 cliff_lower: float = 1.0,
                 cliff_upper: float = 1.0,
                 squared: bool = False,
                 p: float = 2.0,
                 similarity_gate: bool = False,
                 similarity_neg: float = 0.8,
                 similarity_pos: float = 0.2,
                 dev_mode: bool = True):
        super(ACALoss, self).__init__(alpha)
        self.alpha = alpha
        self.cliff_lower = cliff_lower
        self.cliff_upper = cliff_upper
        self.squared = squared
        self.p = p
        self.similarity_gate = similarity_gate
        self.similarity_neg = similarity_neg
        self.similarity_pos = similarity_pos
        self.dev_mode = dev_mode

    def forward(self,
                labels: Tensor,
                predictions: Tensor,
                embeddings: Tensor,
                fingerprints: list = None) -> Tensor:
        """
        Compute the ACA loss.

        Args:
            labels: [B] tensor of true values.
            predictions: [B] tensor of predicted values.
            embeddings: [B, E] tensor of latent embeddings.
            fingerprints: list of length B, each an RDKit ExplicitBitVect.
                          Required if similarity_gate=True; otherwise can be None.

        Returns:
            If dev_mode=True: (loss, reg_loss, tsm_loss, N_Y_ACTs, N_S_ACTs, N_ACTs, N_HV_ACTs)
            If dev_mode=False: loss
        """
        return _aca_loss(
            labels=labels,
            predictions=predictions,
            embeddings=embeddings,
            fingerprints=fingerprints,
            alpha=self.alpha,
            cliff_lower=self.cliff_lower,
            cliff_upper=self.cliff_upper,
            similarity_gate=self.similarity_gate,
            similarity_neg=self.similarity_neg,
            similarity_pos=self.similarity_pos,
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


def get_structure_mask(fingerprints: list,
                       device: torch.device,
                       similarity_neg: float = 0.8,
                       similarity_pos: float = 0.2,
                       eps: float = 1e-5) -> torch.Tensor:
    """
    构造基于指纹（fingerprint）相似度的三元组掩码：
    - fingerprints: 长度为 B 的 Python list，每个元素都是 RDKit ExplicitBitVect
    - similarity_neg: Tanimoto 阈值，要求 sim(a, j) > similarity_neg 才算“正样本候选”
    - similarity_pos: Tanimoto 阈值，要求 sim(a, k) < similarity_pos 才算“负样本候选”
    返回：布尔型张量 mask，形状 [B, B, B]
    """
    B = len(fingerprints)

    # 1. 初始化相似度矩阵 [B, B]
    sim_matrix = torch.zeros((B, B), dtype=torch.float32, device=device)

    # 2. 双重循环计算 Tanimoto 相似度
    for i in range(B):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, B):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    # 3. 构造 “正样本候选” 与 “负样本候选” 的 2D 掩码
    sim_hard_pos = sim_matrix > similarity_neg  # [B, B]: i→j Tanimoto > threshold → 可做“硬正”
    sim_hard_neg = sim_matrix < similarity_pos  # [B, B]: i→k Tanimoto < threshold → 可做“硬负”

    # 4. 扩展到 3D：dim=1 对应 j，dim=2 对应 k
    pos_3d = sim_hard_pos.unsqueeze(2)  # [B, B, 1]
    neg_3d = sim_hard_neg.unsqueeze(1)  # [B, 1, B]
    mask = pos_3d & neg_3d              # [B, B, B]

    # 5. 再加上“索引两两不相等”的约束，保证 i, j, k 三者不同
    idx = torch.arange(B, device=device)
    eq = idx.unsqueeze(0) == idx.unsqueeze(1)  # [B, B]
    neq = ~eq                                  # [B, B]
    i_ne_j = neq.unsqueeze(2)                # [B, B, 1]
    i_ne_k = neq.unsqueeze(1)                # [B, 1, B]
    j_ne_k = neq.unsqueeze(0)                # [1, B, B]
    distinct_idx = i_ne_j & i_ne_k & j_ne_k   # [B, B, B]

    final_mask = mask & distinct_idx          # [B, B, B]
    return final_mask  # 布尔型张量


def _aca_loss(labels: torch.Tensor,
              predictions: torch.Tensor,
              embeddings: torch.Tensor,
              fingerprints: list,
              alpha: float = 0.1,
              cliff_lower: float = 0.2,
              cliff_upper: float = 1.0,
              similarity_gate: bool = False,
              similarity_neg: float = 0.8,
              similarity_pos: float = 0.2,
              squared: bool = False,
              p: float = 2.0,
              dev_mode: bool = True,
              **kwargs):
    """
    ACALoss 的核心实现，结合回归损失与三元组“软 margin”对比损失：
    - labels:     [B] 张量，真实活性
    - predictions:[B] 张量，模型预测活性
    - embeddings: [B, E] 张量，模型输出的潜空间向量
    - fingerprints: 长度为 B 的 Python list，每个元素是 RDKit 指纹 (ExplicitBitVect)
    - alpha: ACA 权重
    - cliff_lower / cliff_upper: 活性差阈值
    - similarity_gate: 是否启用结构相似度过滤
    - similarity_neg / similarity_pos: 结构过滤的上下阈值
    - squared: 是否使用平方回归误差 (MSE)；否则使用 MAE
    - p: cdist 的 p 范数
    - dev_mode: 如果 True，则返回额外的统计信息
    返回:
      如果 dev_mode=True，返回 (loss, reg_loss, tsm_loss, N_Y_ACTs, N_S_ACTs, N_ACTs, N_HV_ACTs)
      否则只返回 loss
    """
    device = embeddings.device
    B = labels.shape[0]

    # 1. 计算回归损失（MAE 或 MSE）
    if squared:
        reg_loss = torch.mean((labels - predictions).abs() ** 2)
    else:
        reg_loss = torch.mean((labels - predictions).abs())

    # 2. 计算标签之间的成对距离，用于“软 margin”
    #    labels_dist: [B, B] 或 [B, B] 的平方距离
    labels_dist = pairwise_distance(labels.unsqueeze(1), p=p, squared=squared)  # [B, B]
    margin_pos = labels_dist.unsqueeze(2)  # [B, B, 1]
    margin_neg = labels_dist.unsqueeze(1)  # [B, 1, B]
    margin = margin_neg - margin_pos       # [B, B, B]

    # 3. 计算 embeddings (潜空间) 之间的成对距离
    pairwise_dis = pairwise_distance(embeddings, p=p, squared=squared)  # [B, B]
    anchor_positive_dist = pairwise_dis.unsqueeze(2)  # [B, B, 1]
    anchor_negative_dist = pairwise_dis.unsqueeze(1)  # [B, 1, B]
    triplet_loss_matrix = anchor_positive_dist - anchor_negative_dist + margin  # [B, B, B]

    # 4. 构造“标签过滤”掩码 (mask_by_y)
    mask_by_y = get_label_mask(labels=labels,
                               device=device,
                               cliff_lower=cliff_lower,
                               cliff_upper=cliff_upper).float()  # [B, B, B]
    N_Y_ACTs = torch.sum(mask_by_y)  # 仅根据标签挖到的三元组总数

    # 5. 如果开启结构过滤 (similarity_gate)，构造“结构过滤”掩码 (mask_by_s)，否则将其设为全 1
    if similarity_gate:
        mask_by_s = get_structure_mask(fingerprints=fingerprints,
                                       device=device,
                                       similarity_neg=similarity_neg,
                                       similarity_pos=similarity_pos).float()  # [B, B, B]
    else:
        # 如果不做结构过滤，则 mask_by_s 全为 1，意味着 “不动它”
        mask_by_s = torch.ones_like(mask_by_y)  # [B, B, B]

    # 5.1 统计仅结构过滤时可挖到的三元组数量（开发时用，dev_mode=False 可不关心）
    N_S_ACTs = torch.sum(mask_by_s)  # 仅根据结构过滤时的三元组数量

    # 6. 将标签过滤和结构过滤结合，得到最终挖到的三元组掩码
    mask_full = (mask_by_y & mask_by_s).float()  # [B, B, B]
    N_ACTs = torch.sum(mask_full)  # 已实际挖到的三元组数量

    # 7. 根据掩码计算 TSM 损失
    if N_ACTs == 0:
        # 没有任何三元组 => 只用回归损失
        tsm_loss = torch.tensor(0.0, device=device)
        loss = reg_loss
        # 如果 dev_mode=True，则 N_HV_ACTs 也设为 0
        N_HV_ACTs = torch.tensor(0.0, device=device)
    else:
        # 先把不需要的 triplet_loss 项目置零，再取正部分之和
        triplet_loss_masked = mask_full * triplet_loss_matrix  # [B, B, B]
        triplet_loss_masked = torch.maximum(triplet_loss_masked,
                                            torch.tensor(0.0, device=device))
        tsm_loss = torch.sum(triplet_loss_masked) / N_ACTs
        loss = reg_loss + alpha * tsm_loss

        # 统计“正的三元组损失项”的个数（即“high-value active triplets”）
        pos_triplets = (triplet_loss_masked > 1e-16).float()
        N_HV_ACTs = torch.sum(pos_triplets)

    # 8. 返回结果
    if dev_mode:
        return loss, reg_loss, tsm_loss, N_Y_ACTs, N_S_ACTs, N_ACTs, N_HV_ACTs
    else:
        return loss
    

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


# ——————————————————————————————————————————————————————————————————————————————
def get_best_structure(fingerprints: list,
                       neg_thresholds: list = list(torch.arange(0.5, 1.0, 0.05).tolist()),
                       pos_thresholds: list = list(torch.arange(0.0, 0.5, 0.05).tolist()),
                       device: torch.device = torch.device('cpu')):
    """
    Get the best (similarity_neg, similarity_pos) that maximizes the number of triplets
    mined purely based on结构相似度条件。

    Args:
        fingerprints: 长度 B 的 Python list，每个元素是 RDKit ExplicitBitVect 指纹。
        neg_thresholds: Tanimoto 阈值列表，用于“硬正/硬负” (sim > neg_threshold)。
        pos_thresholds: Tanimoto 阈值列表，用于“硬负” (sim < pos_threshold)。
        device: torch.device

    Returns:
        best_neg: float, 最佳的 similarity_neg
        best_pos: float, 最佳的 similarity_pos
        df: DataFrame 包含每对 (neg_threshold, pos_threshold, n_triplets) 的列表
    """
    B = len(fingerprints)
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
                fingerprints=fingerprints,
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
            fps = data.fp
            neg, pos, _ = get_best_structure(
                fingerprints=fps,
                neg_thresholds=neg_thresholds,
                pos_thresholds=pos_thresholds,
                device=device
            )
            best_pairs.append((neg, pos))

    df_pairs = pd.DataFrame(best_pairs, columns=['neg', 'pos'])
    neg_mode = df_pairs['neg'].mode().iloc[0]
    pos_mode = df_pairs[df_pairs['neg'] == neg_mode]['pos'].mode().iloc[0]

    return neg_mode, pos_mode
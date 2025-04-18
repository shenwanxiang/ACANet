import sys
# sys.path.insert(0, '/home/shenwanxiang/Research/bidd-clsar/')
sys.path.insert(0, '/mnt/cc/0_ACANet/ACANet')

import os
import math
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
##########################
# 1. 一些辅助函数
##########################
def y_to_pIC50(y):
    """转换 y 到 pIC50"""
    return -np.log10((10**-y)*1e-9)

def compute_scaffolds(smiles_list):
    """计算 Murcko Scaffolds，并返回去重后的数量"""
    scaff_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaff_list.append("InvalidMol")
        else:
            scf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            scaff_list.append(scf)
    return len(set(scaff_list)), list(set(scaff_list))

def compute_mean_tanimoto(smiles_list, fingerprint_type="RDK"):
    """
    计算所有分子两两 Tanimoto 相似度的平均值。
    如果分子特别多，O(N^2) 可能开销很大，注意做抽样/并行/上三角等优化。
    """
    # 先把分子转指纹
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            continue
        if fingerprint_type == "RDK":
            fp = Chem.RDKFingerprint(mol)
        elif fingerprint_type == "Morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        else:
            raise ValueError("Unknown fingerprint_type")
        fps.append(fp)
    
    sim_list = []
    N = len(fps)
    for i in range(N):
        fp_i = fps[i]
        if fp_i is None:
            continue
        for j in range(N):
            fp_j = fps[j]
            if fp_j is None:
                continue
            sim_list.append(DataStructs.TanimotoSimilarity(fp_i, fp_j))
    
    if len(sim_list) == 0:
        return float('nan')
    return np.mean(sim_list)

##########################
# 2. 多进程要执行的单一函数
##########################
def process_single_dataset(dataset_name):
    """
    子进程要执行的工作函数：
      - 读取指定数据集 CSV
      - 提取 smiles, y
      - 计算 mean_sim, scaffold_num
      - 返回必要的信息供主进程汇总与画图
    """
    csv_path = f'/mnt/cc/0_ACANet/ACANet/experiment/05_Review/MPCD/dataset/LSSNS/{dataset_name}.csv'
    if not os.path.exists(csv_path):
        print(f"[Warning] File not found: {csv_path}")
        # 返回空结果，主进程自行忽略
        return (dataset_name, None, None, [], [])

    df = pd.read_csv(csv_path)
    smiles_list = df['Smiles'].tolist()
    y_raw = df['pChEMBL Value'].values

    # 计算 scaffold 数量
    num_scaffolds, scaffolds_list = compute_scaffolds(smiles_list)

    # 计算平均 Tanimoto
    mean_sim = compute_mean_tanimoto(smiles_list, fingerprint_type="RDK")

    mean_sim_scaffolds =  compute_mean_tanimoto(scaffolds_list, fingerprint_type="RDK")

    # 返回 (dataset_name, mean_sim, num_scaffolds, y_values, smiles_list 等等)
    return (dataset_name, mean_sim, mean_sim_scaffolds, num_scaffolds, y_raw, smiles_list)

##########################
# 3. 主程序
##########################
if __name__ == "__main__":
    # 3.1 准备数据集列表
    url = 'https://raw.githubusercontent.com/bidd-group/MPCD/main/dataset/LSSNS/'
    datasets = ['BRAF','IDO1', 'PHGDH', 'PKCi', 'PLK1','RIP2', 'RXFP1', 'USP7', 'mGluR2']
    path = '/mnt/cc/0_ACANet/ACANet/experiment/05_Review/MPCD/dataset/LSSNS'
    save_dir = 'review_result'
    

    # 3.2 并行处理
    n_processes = 4  # 你可以根据CPU核心数调整
    with Pool(n_processes) as p:
        results = []
        for res in tqdm(p.imap(process_single_dataset, datasets), total=len(datasets)):
            results.append(res)

    # 3.3 处理结果 (有些可能返回 None)
    #     results 是个 list，每个元素是子进程返回的 tuple
    #     (dataset_name, mean_sim, num_scaffolds, y_values, smiles_list)
    filtered_results = [r for r in results if r[1] is not None]  # 排除没读到文件的

    # 3.4 保存统计到 CSV
    #     并把要画图的 y 也缓存一份
    summary = []
    dataset_to_y = {}
    for (dataset_name, mean_sim, mean_sim_scaffolds, num_scaffolds, y_raw, _) in filtered_results:
        summary.append([dataset_name, mean_sim, num_scaffolds])#, mean_sim_scaffolds])
        dataset_to_y[dataset_name] = y_raw

    df_summary = pd.DataFrame(summary, columns=["dataset_name", "mean_tanimoto", "num_scaffolds"])#, "mean_scaffolds_tanimoto"])
    os.makedirs("review_result", exist_ok=True)
    out_csv = os.path.join("review_result", "LSSNS_sim_scaffold_summary.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"[Done] 写入结果到 {out_csv}")

    # 3.5 一次性画所有数据集的 y 分布图
    ncols = 3
    nrows = math.ceil(len(filtered_results) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = axes.flatten()

    for idx, (dataset_name, mean_sim, mean_sim_scaffolds, num_scaffolds, y_raw, _) in enumerate(filtered_results):
        ax = axes[idx]
        sns.histplot(y_raw, ax=ax, kde=True, color='skyblue')
        ax.set_title(f"{dataset_name}")
        ax.set_xlabel("pChEMBL Value")

    # 隐藏多余子图
    for extra_idx in range(len(filtered_results), len(axes)):
        axes[extra_idx].axis('off')

    plt.tight_layout()
    png_path = os.path.join("review_result", "LSSNS_y_distributions_all_datasets.pdf")
    plt.savefig(png_path, dpi=2500)
    plt.show()
    print(f"[Done] 画图完成，保存到 {png_path}")

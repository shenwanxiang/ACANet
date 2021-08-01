# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 22:02:01 2021

@author: wanxiang.shen@u.nus.edu

"""

from scipy.spatial.distance import squareform 
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from itertools import chain
import numpy as np
import pandas as pd  
from tqdm import tqdm
from joblib import dump, load
import os
tqdm.pandas(ascii=True)


## offline triplet mining
## online triplet hard loss: https://github.com/bidd-group/pytorch-TripletSemiHardLoss
class Sampling:
    
    def __init__(self, input_file = './input.csv'):
        self.df = pd.read_csv(input_file, header=None)
        assert self.df.shape[1] == 2, 'input file should contain two columns: smiles and labels'
        
        self.arr_smiles = df[0].values
        self.df_dist = pd.DataFrame(self.pairwise_dist(self.arr_smiles))
        
        group = self.df.groupby(1).apply(lambda x:x.index.tolist())
        self.pos_idx = group[1]
        self.neg_idx = group[0]

        
    def _calc_ecfp4(self, smiles):
        ecfp4 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smiles), radius = 2)    
        return ecfp4

    
    def pairwise_dist(self, smiles_list):    
        MorganFP_list = [self._calc_ecfp4(i) for i in smiles_list1]
        TanimotoDist =[]   
        for i, fp1 in tqdm(enumerate(MorganFP_list), ascii=True):
            for fp2 in MorganFP_list[i+1:]:
                s = DataStructs.TanimotoSimilarity(fp1,fp2)
                #Available similarity metrics include Tanimoto, Dice, 
                # Cosine, Sokal, Russel, Kulczynski, McConnaughey, and Tversky
                d = 1. - s #distance
                TanimotoDist.append(d)
        dist_matrix = squareform(TanimotoDist)
        return dist_matrix

    def triplet_mining(self,  ap_dist_gt = 0.6, np_dist_ls = 0.4):
        '''
        ap_dist_gt: criteria to get the hard negatives, they will be selected if their dist > ap_dist_gt
        np_dist_ls: criteria to get the hard negatives, they will be selected if their dist < np_dist_ls
        '''

        arr_smiles = df[0].values
        #hard_postive
        ap_pairs = []
        for i in range(len(group[1])):
            a = group[1][i]
            for p in group[1][i+1:]:
                ap_dist = dist_matrix[a][p]
                if ap_dist > ap_dist_gt:
                    ap = [a, p]
                    ap_pairs.append(ap)
                    
        #hard_negative
        triplets = []
        for a, p in tqdm(ap_pairs, ascii=True):
            for n in group[0]:
                an_dist = dist_matrix[a][n]
                pn_dist = dist_matrix[p][n]
                if (an_dist < np_dist_ls) | (pn_dist < np_dist_ls):
                    triplets.append([a,p,n])
        return triplets
    

    def duplet_mining(self, np_dist_ls = 0.4):
        '''
        np_dist_ls: criteria to get the hard negatives, they will be selected if their dist < 0.4
        '''
        p_n_dist_matrix = self.df_dist.iloc[self.pos_idx][self.neg_idx]
        mask = p_n_dist_matrix < np_dist_ls

        duplets = mask.progress_apply(lambda x: [[x.name, i] for i in x[x].index], axis=1)
        duplets = list(chain(*duplets))
        
        return duplets

    
if __name__ == '__main__':
    
    S = Sampling()

    ## get indices
    duplet_idx = S.duplet_mining()
    triplet_idx = S.triplet_mining()
    
    ## get smiles
    for triplet in triplet_idx:
        tri_smiles = S.arr_smiles[triplet]

    for duplet in duplet_idx:
        dup_smiles = S.arr_smiles[duplet]
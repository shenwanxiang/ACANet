import sys
sys.path.insert(0, '/home/shenwanxiang/Research/bidd-clsar/')
from clsar import ACANet
import pandas as pd
import numpy as np
import os, json

url = 'https://raw.githubusercontent.com/bidd-group/MPCD/main/dataset/LSSNS/'
datasets = ['BRAF','IDO1', 'PHGDH', 'PKCi', 'PLK1','RIP2', 'RXFP1', 'USP7', 'mGluR2']




for dataset_name in datasets:
    
    save_dir = './benchmark_performance/mae_aca_opt_cliff/%s' % dataset_name
    
    dataset_file = os.path.join(url, dataset_name + '.csv')
    df = pd.read_csv(dataset_file)

    X = df.Smiles.values
    y = df['pChEMBL Value'].values

    clf = ACANet(gpuid = 0,  squared=False,  work_dir = save_dir)
    #dfp = clf.opt_cliff_by_trps(X, y, iterations=5)
    
    ## 5FCV performance
    dfp = clf.opt_cliff_by_cv(X, y, total_epochs=800, n_repeats=3)
    dfa = clf.opt_alpha_by_cv(X, y, total_epochs=800, n_repeats=10, save_rawdata=True)

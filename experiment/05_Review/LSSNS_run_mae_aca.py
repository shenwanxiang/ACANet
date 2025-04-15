import sys
sys.path.insert(0, '/mnt/cc/0_ACANet/ACANet')
from clsar import ACANet
import pandas as pd
import numpy as np
import os, json
from clsar.feature import Gen39AtomFeatures,  Gen39AtomFeatures_full
url = 'https://raw.githubusercontent.com/bidd-group/MPCD/main/dataset/LSSNS/'
datasets = ['BRAF','IDO1', 'PHGDH', 'PKCi', 'PLK1','RIP2', 'RXFP1', 'USP7', 'mGluR2']




for dataset_name in datasets:
    path = '/mnt/cc/0_ACANet/ACANet/experiment/05_Review/MPCD/dataset/LSSNS'
    save_dir = './benchmark_performance/mae_aca_opt_cliff/%s' % dataset_name
    
    dataset_file = os.path.join(path, dataset_name + '.csv')
    df = pd.read_csv(dataset_file)

    X = df.Smiles.values
    y = df['pChEMBL Value'].values

    clf = ACANet(gpuid = 0,  squared=False,  pre_transform=Gen39AtomFeatures_full(), work_dir = save_dir,fp_filter=True, scaffold_filter=True)
    #dfp = clf.opt_cliff_by_trps(X, y, iterations=5)
    
    ## 5FCV performance
    # dfp = clf.opt_cliff_by_cv(X, y, total_epochs=5, n_repeats=1)
    # dfa = clf.opt_alpha_by_cv(X, y, total_epochs=5, n_repeats=1, save_rawdata=True)
    clf.cv_fit_dev(X, y, verbose=1, dev_mode=True)
import sys
# sys.path.insert(0, '/home/shenwanxiang/Research/bidd-clsar/')
sys.path.insert(0, '/mnt/cc/0_ACANet/ACANet')
from clsar import ACANet
import pandas as pd
import numpy as np
import os, json
from clsar.feature import Gen39AtomFeatures,  Gen39AtomFeatures_full
meta = 'https://bidd-group.github.io/MPCD/dataset/HSSMS/MoleculeACE_benchmark/metadata/datasets.csv'
meta_table = pd.read_csv(meta)
datasets = meta_table.Dataset.tolist()     


def y_to_pIC50(y):
    pIC50 = -np.log10((10**-y)*1e-9)
    return pIC50


def pIC50_to_y(pIC50):
    y = -np.log10(10**-pIC50*1e9)
    return y


for dataset_name in ['CHEMBL4203_Ki']: #['CHEMBL4203_Ki']:
    save_dir = './benchmark_performance/mae_aca_opt_cliff/%s' % dataset_name
    df = pd.read_csv('/mnt/cc/0_ACANet/ACANet/experiment/05_Review/MPCD/dataset/HSSMS/MoleculeACE_benchmark/%s.csv' % dataset_name)
    
    df_train = df[df.split == 'train']
    df_test = df[df.split == 'test']
    Xs_train = df_train.smiles.values
    y_train = df_train.y.values
    ## convert y to pIC50
    y_train_pIC50 = y_to_pIC50(y_train)

    ## get loss parameters by training set
    clf = ACANet(gpuid = 0,   work_dir = save_dir, pre_transform=Gen39AtomFeatures_full(), scaffold_filter=True)
    #dfp = clf.opt_cliff_by_trps(Xs_train, y_train_pIC50, iterations=5)
    dfp = clf.opt_cliff_by_cv(Xs_train, y_train_pIC50, total_epochs=50, n_repeats=3)
    dfa = clf.opt_alpha_by_cv(Xs_train, y_train_pIC50, total_epochs=100, n_repeats=3)
    # dfp = clf.opt_cliff_by_cv(Xs_train, y_train_pIC50, total_epochs=5, n_repeats=1)
    # dfa = clf.opt_alpha_by_cv(Xs_train, y_train_pIC50, total_epochs=5, n_repeats=1)
    
    ## 5FCV fit to generate 5 sub-models, selection best model by the model performance of the val set
    clf.cv_fit_dev(Xs_train, y_train_pIC50, verbose=1, dev_mode=True)
    
    ## 5FCV predict and convert pIC50 to y
    test_pred_pIC50 = clf.cv_predict(df_test.smiles)
    test_pred = pIC50_to_y(test_pred_pIC50)

    
    test_true = df_test.y.values
    cliff_flag = df_test.cliff_mol.values
    dfp = pd.DataFrame([test_true, test_pred, cliff_flag]).T
    dfp.columns = ['y_true', 'y_pred', 'cliff_flag']

    ## calculate the overall rmse and cliff rmse of the test set
    overall_rmse = np.sqrt(np.mean(np.square(dfp.y_true - dfp.y_pred)))
    g = dfp.groupby('cliff_flag')
    noncliff_rmse, cliff_rmse = g.apply(lambda x:np.sqrt(np.mean(np.square(x.y_true - x.y_pred)))).to_list()
    results = {'dataset':dataset_name, 'rmse':overall_rmse, 'cliff_rmse':cliff_rmse}

    ## save the results
    file2save = os.path.join(save_dir,  'results.json')
    with open(file2save, 'w') as fp:
        json.dump(results, fp)

# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:34:56 2022

@author: wanxiang.shen@u.nus.edu
"""


import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
from tqdm import tqdm
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os, json
import seaborn as sns
sns.set(style='white',  font='sans-serif', font_scale=2)


from clsar.feature import Gen39AtomFeatures  # feature
from clsar.model.model import ACANet_PNA, get_deg, _fix_reproducibility # model
from clsar.model.loss import ACALoss, get_best_cliff
from clsar.model.saver import SaveBestModel
from clsar.model.train import train, test, predict


class ACANet:

    def __init__(self,
                 
                 #training control parameters
                 epochs = 800,
                 batch_size = 128,
                 lr = 1e-4, 
                 
                 ## loss parameters
                 alpha = 1e-1,
                 cliff_lower = 1.0,
                 cliff_upper = 1.0,
                 squared = False,
                 p = 2.0,
                 
                 #feature parameters
                 pre_transform = Gen39AtomFeatures(),
                 
                 # model paramaters
                 out_channels = 1,  #output dim.
                 convs_layers = [64, 128, 256, 512],  
                 dense_layers = [256, 128, 32], 
                 pooling_layer = global_max_pool,
                 batch_norms = torch.nn.BatchNorm1d,
                 dropout_p = 0.0,
                 aggregators = ['mean', 'min', 'max', 'sum', 'std'],
                 scalers = ['identity', 'amplification', 'attenuation'],
                 
                 
                 ## others
                 gpuid = 0,
                 work_dir = './',
                 **model_args):
        
        
        
        '''A High-level packaging of the ACANet model for easier use.

        Parameters
        --------------------------
        
        aggregators (List[str]) – Set of aggregation function identifiers, namely "sum", "mean", "min", "max", "var" and "std".

        scalers (List[str]) – Set of scaling function identifiers, namely "identity", "amplification", "attenuation", "linear" and "inverse_linear".

        '''
        super().__init__()
        
        _fix_reproducibility(42)
        
        #training control parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        ## loss parameters
        self.alpha = alpha
        self.cliff_lower = cliff_lower
        self.cliff_upper = cliff_upper        
        self.squared = squared
        self.p = p
        
        ## feature parameters
        self.pre_transform = pre_transform
        self.in_channels = pre_transform.in_channels # node channel
        self.edge_dim = pre_transform.edge_dim # edge channel
        
        
        ## model parameters
        self.out_channels = out_channels
        self.convs_layers = convs_layers
        self.dense_layers = dense_layers
        self.pooling_layer = pooling_layer
        self.batch_norms = batch_norms
        self.dropout_p = dropout_p
        self.aggregators = aggregators
        self.scalers = scalers
        
            
        self.gpuid = gpuid
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        
        torch.cuda.set_device(gpuid)
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_pub_args = { 'in_channels': self.in_channels, 
                                'out_channels': self.out_channels, 
                                 'edge_dim': self.edge_dim, 
                                 'convs_layers': self.convs_layers, 
                                 'dense_layers': self.dense_layers,
                                 'dropout_p':self.dropout_p, 
                                 'pooling_layer':self.pooling_layer,
                                 'batch_norms':self.batch_norms,
                               'aggregators':self.aggregators,
                               'scalers':self.scalers,
                              
                              } 

        self.model_pub_args.update(model_args)
        self.cv_models = []

        
    def _setup(self, train_dataset, alpha, cliff_lower, cliff_upper):
        
        '''
        To setup the PNN-ACANet with ACA loss
        '''
        deg = get_deg(train_dataset)
        
        model = ACANet_PNA(**self.model_pub_args, deg=deg, ).to(self.device)
        
        aca_loss = ACALoss(alpha = alpha, 
                           cliff_lower = cliff_lower, 
                           cliff_upper = cliff_upper, 
                           dev_mode=False, 
                           p = self.p, 
                           squared = self.squared)
        
        saver = SaveBestModel(data_transformer = self.smiles_to_data, 
                              save_dir = self.work_dir, 
                              save_name = 'best_model.pth')

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=10**-5) #
        return model, optimizer, aca_loss, saver
    


    def smiles_to_data(self, smiles_list):
        data_list = [] 
        for smiles in smiles_list:
            data = self.pre_transform(Data(smiles=smiles)) 
            data_list.append(data)
        return data_list
    
    
    
    def _Xy_to_dataset(self, Xs, y):
        dataset = []
        for smi, _y in zip(Xs, y):
            y_tensor = torch.tensor([_y], dtype=torch.float).view(1, -1)
            data = self.pre_transform(Data(smiles=smi, y = y_tensor)) 
            dataset.append(data)
        return dataset
        

    def _cv_split(self, y_train, n_folds = 5, random_state = 42):
        
        if len(y_train) >= 250:
            ss = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
            cutoff = np.median(y_train)
            labels = [0 if i < cutoff else 1 for i in y_train]
            splits = [{'inner_train_idx': i, 'inner_val_idx': j} for i, j in ss.split(labels, labels)]
        else:
            idx = np.argsort(y_train,axis=0).reshape(-1,) #sort data by ther value
            ts = pd.Series(idx)
            splits = []
            for i in range(n_folds):
                ts_idx = ts.iloc[i::n_folds].tolist()
                tr_idx = ts[~ts.isin(ts_idx)].tolist()
                splits.append({'inner_train_idx':tr_idx, 'inner_val_idx':ts_idx})
                
        return splits


    def _cv_performance(self, 
                        Xs_train, 
                        y_train,
                        alpha,
                        cliff_lower, 
                        cliff_upper, 
                        n_folds = 5, 
                        total_epochs = 300,
                        random_state = 42,
                       ):


        splits = self._cv_split(y_train, n_folds = n_folds, random_state = random_state)
        train_dataset = self._Xy_to_dataset(Xs_train, y_train)
        
        _performance = []
        for i_split, split in enumerate(splits):
            inner_train = pd.Series(train_dataset).iloc[split['inner_train_idx']].to_list()
            inner_val = pd.Series(train_dataset).iloc[split['inner_val_idx']].to_list()

            inner_train_loader = DataLoader(inner_train, batch_size=self.batch_size, shuffle=True)
            inner_val_loader = DataLoader(inner_val, batch_size=self.batch_size)
            
            model, optimizer, aca_loss, saver = self._setup(train_dataset, alpha = alpha, 
                                                           cliff_lower = cliff_lower, 
                                                            cliff_upper = cliff_upper)

            history = []
            for epoch in tqdm(range(total_epochs), desc='epoch', ascii=True):
                _ = train(inner_train_loader, model, optimizer, aca_loss, self.device)
                val_rmse = test(inner_val_loader, model, self.device)
                history.append(val_rmse)
                #saver(val_rmse, epoch, model, optimizer)

            ts = pd.Series(history)
            ts.index = ts.index + 1
            ts = ts.to_frame(name = i_split)
            _performance.append(ts)
        return pd.concat(_performance, axis=1)


    
    def opt_cliff_by_trps(self, Xs_train, y_train, iterations = 5):
        '''
        Get the best cliff parameter via the number of the triplets that mined.
        '''
        vmin = 0.0
        vmax = 4.1
        cliffs = list(np.arange(vmin, vmax, 0.1).round(2))
        train_dataset = self._Xy_to_dataset(Xs_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        print('Get the best cliff parameter via the number of the triplets that mined...')
        c = []
        dfcs = []
        for epoch in tqdm(range(iterations), desc = 'epoch', ascii=True):
            for data in train_loader:
                cl, cu, dfc = get_best_cliff(data.y.to(self.device), cliffs = cliffs)
                c.append([cl, cu, epoch])
                dfcs.append(dfc)
        c_distribution = pd.DataFrame(c).groupby(0).size()/len(c)
        best_cliff = c_distribution.idxmax()
        self.cliff_lower = best_cliff
        self.cliff_upper = best_cliff
        print('Best cliff_lower/cliff_upper by counting the number of the mined triplet is: %s' % best_cliff)

        dfp = sum(dfcs)/len(dfcs)
        plot_dfp_save(dfp, save_dir = self.work_dir)
        return dfp

        
    
    def opt_alpha_by_cv(self, Xs_train, y_train, n_folds = 5, n_repeats=1, total_epochs=800, save_rawdata=False, alphas = [0., 0.01,  0.05,  0.1 , 0.2,  0.5  , 1.   ], random_state = 42):
        '''
        Get the best alpha parameter by the cross-validation performance
        '''

        w = 5
        res = []
        rawdata = []
        for alpha in alphas:
            rmses = []
            for rp in range(n_repeats):
                dfcv = self._cv_performance(Xs_train, y_train,
                                       alpha = alpha,
                                       cliff_lower = self.cliff_lower, 
                                       cliff_upper = self.cliff_upper, 
                                       n_folds = n_folds, 
                                       total_epochs = total_epochs, random_state = random_state)
                
                rawdata.append({'alpha':alpha, 'repeat':rp, 'dfcv':dfcv})
                rmse_best = dfcv.mean(axis=1).rolling(w).mean().min()
                rmses.append(rmse_best)

            rmse = np.mean(rmses)
            rmse_err = np.std(rmses)
            res.append([alpha, rmse, rmse_err])

        if save_rawdata:
            dump(rawdata, os.path.join(self.work_dir, 'alpha_performance_rawdata.pkl'))
        
        dfa = pd.DataFrame(res, columns=['alpha', 'rmse', 'rmse_err'])
        best_alpha = dfa.iloc[dfa.rmse.idxmin()].alpha
        self.alpha = best_alpha
        print('Best cliff-awareness factor alpha by cross-validation is: %s' % best_alpha)
        
        plot_dfa_save(dfa, save_dir = self.work_dir)
       
        
        return dfa
    

    
    def opt_cliff_by_cv(self, Xs_train, y_train, n_folds = 5, n_repeats=1, total_epochs = 800,  upper_cliffs = list(np.arange(0.5, 4.5, 0.5).round(2)), random_state = 42):

        #upper_cliffs = list(np.arange(0.3, 4.2, 0.3).round(2))
       
        # find best cliff_upper
        w = 3
        res = []
        for cliff_upper in upper_cliffs:
            cliff_lower = 0.1
            
            rmses = []
            for rp in range(n_repeats):
                dfcv = self._cv_performance(Xs_train, y_train,
                                       alpha = self.alpha,
                                       cliff_lower = cliff_lower, 
                                       cliff_upper = cliff_upper, 
                                       n_folds = n_folds, 
                                       total_epochs = total_epochs, random_state = random_state)
                rmse_best = dfcv.mean(axis=1).rolling(w).mean().min()
                rmses.append(rmse_best)
 
            rmse = np.mean(rmses)
            rmse_err = np.std(rmses)
            res.append([cliff_lower, cliff_upper, rmse, rmse_err])
        
        df1 = pd.DataFrame(res, columns=['cl', 'cu', 'rmse', 'rmse_err'])

        #idxmin = df1.rmse.rolling(2).mean().idxmin()
        idxmin = df1.rmse.idxmin()
        best_cliff_upper = df1.iloc[idxmin].cu

        # find best cliff_lower
        res = []
        start = 0.1
        stop = best_cliff_upper
        cs = np.arange(start, stop, (stop-start)/8)
        lower_cliffs = list(cs.round(2))
        for cliff_lower in lower_cliffs:
            cliff_upper = best_cliff_upper
            
            rmses = []
            for rp in range(n_repeats):
                dfcv = self._cv_performance(Xs_train, y_train,
                                       alpha = self.alpha,
                                       cliff_lower = cliff_lower, 
                                       cliff_upper = cliff_upper, 
                                       n_folds = n_folds, 
                                       total_epochs = total_epochs, random_state = random_state)
                rmse_best = dfcv.mean(axis=1).rolling(w).mean().min()
                rmses.append(rmse_best)
                
            rmse = np.mean(rmses)
            rmse_err = np.std(rmses)
            res.append([cliff_lower, cliff_upper, rmse, rmse_err])
        df2 = pd.DataFrame(res, columns=['cl', 'cu', 'rmse', 'rmse_err'])
        dfp = df1.append(df2)
        dfc = dfp.drop_duplicates(['cl', 'cu']).reset_index(drop=True)
        
        best = dfc.iloc[dfc.rmse.idxmin()]
        best_cliff_lower = best.cl
        best_cliff_upper = best.cu
        
        self.cliff_lower = best_cliff_lower
        self.cliff_upper = best_cliff_upper
        print('Best cliff_lower and cliff_upper parameter is: %s and %s, respectively.' % (best_cliff_lower, best_cliff_upper))
        plot_dfc_save(dfc, save_dir = self.work_dir)
        return dfc
   

    def opt_epoch_by_cv(self, Xs_train, y_train, 
                        n_folds = 5, 
                        total_epochs = 800, random_state = 42):
        
        
        dfcv = self._cv_performance(Xs_train, y_train,
                                   alpha = self.alpha,
                                   cliff_lower = self.cliff_lower, 
                                   cliff_upper = self.cliff_upper, 
                                   n_folds = n_folds, 
                                   total_epochs = total_epochs, random_state = random_state)
        dfe = dfcv.mean(axis=1).rolling(100).mean()        
        best_epoch = dfe.idxmin()
        self.epochs = best_epoch
        print('Best epochs by cross-validation is: %s' % best_epoch)
        
        plot_dfe_save(dfe, save_dir = self.work_dir)
        
        return dfe


    def fit(self, Xs_train, y_train, Xs_val = None, y_val = None, verbose = 1, save_model=False):
        
        '''
        Xs_train: list or array of smiles
        y_train: list or array
        '''
        train_dataset = self._Xy_to_dataset(Xs_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if (Xs_val is None) | (y_val is None):
            val_dataset = None
            val_loader = None
        else:
            val_dataset = self._Xy_to_dataset(Xs_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        history = []
        model, optimizer, aca_loss, saver = self._setup(train_dataset, 
                                                        alpha = self.alpha, 
                                                        cliff_lower = self.cliff_lower, 
                                                        cliff_upper = self.cliff_upper)

        for epoch in range(1, self.epochs+1):
            train_loss = train(train_loader, model, optimizer, aca_loss, self.device)
            if val_loader is None:
                val_rmse = np.nan
                saver(train_loss, epoch, model, optimizer)
            else:
                val_rmse = test(val_loader, model, self.device)
                saver(val_rmse, epoch, model, optimizer)
            if verbose:
                print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f} Test: {val_rmse:.4f}')
            history.append({'Epoch': epoch, 'train_loss': train_loss,'val_rmse': val_rmse})

        model.load_state_dict(saver.inMemorySave['model_state_dict'])
        optimizer.load_state_dict(saver.inMemorySave['optimizer_state_dict'])
        model.eval()
        
        self.model = model
        self.optimizer = optimizer
        self.aca_loss = aca_loss
        self.saver = saver
        
        model_args = {}
        for k, v in self.model.model_args.items():
            model_args[k] = str(v)
        config = {'training': {'lr': self.lr, 'epochs': self.epochs,'batch_size': self.batch_size},
                  'loss': {'alpha': self.alpha, 
                           'cliff_lower': self.cliff_lower, 'cliff_upper': self.cliff_upper, 
                           'squared':self.squared, 'p': self.p},
                  'model':model_args}
        
        ## save the config
        file2save = os.path.join(self.work_dir,  'config.json')
        with open(file2save, 'w') as fp:
            json.dump(config, fp)
        
        ## save the model
        if save_model:
            self.saver.save()
        
        return self
       

        
    def predict(self, Xs_test):
        '''
        Make prediction for the list of mols
        
        Parameters
        -----------
        Xs_test: list or array of smiles
        '''
        test_dataset = self.smiles_to_data(Xs_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        test_pred = predict(test_loader, self.model, self.device)
        y_pred = test_pred.cpu().numpy().reshape(-1, )    
        return y_pred

    

    def cv_fit(self, Xs_train, y_train, n_folds = 5, verbose=1, random_state = 42):
        '''
        Fit the model by cross-validation strategy, you will generate the n_fold sub-models for the prediction
        '''

        splits = self._cv_split(y_train, n_folds = n_folds, random_state = random_state)
        cv_models = []
        for i, split in enumerate(splits):
            inner_train_x = Xs_train[split['inner_train_idx']]
            inner_train_y = y_train[split['inner_train_idx']]
            
            inner_val_x = Xs_train[split['inner_val_idx']]
            inner_val_y = y_train[split['inner_val_idx']]
            self.fit(inner_train_x, inner_train_y, 
                     inner_val_x, inner_val_y, 
                     verbose=verbose)
            
            cv_models.append(self.model)
            
        self.cv_models = cv_models
        
        return self
    
    
    def cv_predict(self, Xs_test):
        '''
        Make prediction for the list of mols
        
        Parameters
        -----------
        Xs_test: list or array of smiles
        '''
        
        assert len(self.cv_models) >= 1, 'Please fit the data by fit_cv first!'
        test_dataset = self.smiles_to_data(Xs_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        cv_predictions = []
        for model in self.cv_models:
            test_pred = predict(test_loader, model, self.device)
            y_pred = test_pred.cpu().numpy().reshape(-1, )    
            cv_predictions.append(y_pred)
           
        ## avg. predictions
        y_pred_final = np.mean(cv_predictions, axis=0)
        
        return y_pred_final
      
    
    def save(self, mfile):
        dump(self, mfile)
        print('Saving the model to %s' % mfile)

        
        
    def load(self, mfile):
        self = load(mfile)
        return self
        

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    
    
    
    
    

def plot_dfa_save(dfa, save_dir):

    dfa.alpha = dfa.alpha.apply(lambda x:'%.e' % float(x))
    fig, ax = plt.subplots(figsize=(8, 6))
    #im = ax.scatter(dfa.alpha, dfa.rmse, c=dfa.rmse, cmap='jet', s=200)
    #colors = sns.color_palette('jet_r', len(dfa)).as_hex()

    ax.plot(dfa.alpha, dfa.rmse)
    ax.scatter(dfa.alpha, dfa.rmse, c=dfa.rmse, cmap='rainbow', s=200, alpha = 1)
    ax.errorbar(dfa.alpha, dfa.rmse, yerr=dfa.rmse_err, capsize = 6, ecolor='#8c8c8c', color = '#999999')

    
    ax.set_xlabel('alpha')
    ax.set_ylabel('CV RMSE')

    ax.tick_params(left='off',  bottom='off', pad=.5,)
    ax.set_title('Alpha performance')

    ax.tick_params(axis='x', labelrotation=60)

    fig.savefig(os.path.join(save_dir,'alpha_performance.png'), dpi=300, bbox_inches='tight')
    dfa.to_csv(os.path.join(save_dir,'alpha_performance.csv'))
    

    
def plot_dfc_save(dfc, save_dir):

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.scatter(dfc.cl, dfc.cu, c=dfc.rmse, cmap='jet', s=200)
    ax.set_xlabel('cliff lower')
    ax.set_ylabel('cliff upper')
    cbar2 = fig.colorbar(im, ax=ax, aspect=40, pad=0.02)
    cbar2.set_label('CV RMSE', rotation=90)
    fig.tight_layout()
    ax.tick_params(left='off',  bottom='off', pad=.5,)
    vm = ax.get_ylim()[1]
    ax.set_xlim(0, vm)
    ax.set_ylim(0.1, vm)
    ax.set_title('Cliff performance')

    fig.savefig(os.path.join(save_dir,'cliff_performance.png'), dpi=300, bbox_inches='tight')
    dfc.to_csv(os.path.join(save_dir,'cliff_performance.csv'))
    
    
def plot_dfp_save(dfp1, save_dir):
    
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib import ticker

    dfp1['trps2'] = dfp1.trps.replace(0, np.nan)
    dfp1['trps2'] = np.log2(dfp1['trps2'])
    ticks = list(np.arange(dfp1.lower.min(), dfp1.lower.max() + 0.3, 0.5).round(2))
    v = dfp1.trps2.dropna().sort_values().astype(int)
    bds = [0]

    bds1 = np.linspace(0, v.max(), 20)
    bds.extend(bds1)
    bds = pd.Series(bds).astype(float).to_list()

    base_cmaps = ['Greys', 'gist_ncar_r'] #gist_ncar_r 

    n_base = len(base_cmaps)
    N=[1, len(bds1)]# number of colors  to extract from each cmap, sum(N)=len(classes)
    colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.1, 0.9, N[i])) for i,name in zip(range(n_base),base_cmaps)])
    cmap = ListedColormap(colors)
    boundary_norm = BoundaryNorm(bds, cmap.N)
    fig, ax = plt.subplots(figsize=(9, 7))
    s = 70
    marker = 'o'
    lw = 0

    im = ax.scatter(x = dfp1.lower, 
                     y = dfp1.upper, #vmax = dfp1.trps.max(), 
                     c = np.log2(dfp1.trps+1), 
                     norm = boundary_norm, #marker = ',',
                     marker = marker,
                     edgecolors='k', 
                     lw=lw, 
                     s = s, cmap= cmap, label = 'trps2')

    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    fmt.format = '$\\mathdefault{%1.2f}$'
    cbar = fig.colorbar(im, ax=ax, aspect=40, pad = 0.02, format = fmt,)# 
    cbar.set_label('Log2 No. of mined triplets')

    ax.tick_params(left='off',  bottom='off', pad=.3,)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_title('Cliff vs. Triplets')
    ax.set_ylabel('cliff upper')
    ax.set_xlabel('cliff lower')
    fig.tight_layout()
    
    fig.savefig(os.path.join(save_dir,'triplets_distribution.png'), dpi=300, bbox_inches='tight')
    dfp1.to_csv(os.path.join(save_dir,'triplets_distribution.csv'))
    
    
def plot_dfe_save(dfe, save_dir):

    best_epoch = dfe.idxmin()
    
    vmax = dfe.max()#-1
    vmin = dfe.min()-0.2

    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = dfe.plot()
    ax.set_xlabel('epochs')
    ax.set_ylabel('CV RMSE')
    
    ax.vlines(best_epoch, 0, 100, ls = '--', color = 'red', lw = 2)
    ax.text(best_epoch, 0.0, '%s' % best_epoch, color='red', transform=ax.get_xaxis_transform(),
            ha='center', va='top')
         
    fig.tight_layout()
    ax.tick_params(left='off',  bottom='off', pad=.5,)

    ax.set_ylim(vmin, vmax)
    
    ax.set_title('CV performance')

    fig.savefig(os.path.join(save_dir,'epoch_performance.png'), dpi=300, bbox_inches='tight')
    dfe.to_csv(os.path.join(save_dir,'epoch_performance.csv'))
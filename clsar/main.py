# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:34:56 2022

@author: wanxiang.shen@u.nus.edu
"""

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from torch import Tensor
         
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load

from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

sns.set(style='white', font='sans-serif', font_scale=2)


def get_morgan_fingerprint(mol, radius=2, nBits=2048):
    """Calculates a Morgan fingerprint for a given RDKit Mol."""
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)


# ——————————————————————————————————————————————————————————————————————————————
# 注意：下面这些导入需要在 clsar/model 中已经实现过
from clsar.feature import Gen39AtomFeatures  # node/edge feature transformer
from clsar.model.model import ACANet_PNA, get_deg, _fix_reproducibility
from clsar.model.loss import ACALoss, get_best_cliff, get_best_structure_threshold
from clsar.model.saver import SaveBestModel
from clsar.model.train import train, test, predict


class ACANet:

    def __init__(self,
                 # training control parameters
                 epochs: int = 800,
                 batch_size: int = 128,
                 lr: float = 1e-4,

                 # loss parameters
                 alpha: float = 1e-1,
                 cliff_lower: float = 1.0,
                 cliff_upper: float = 1.0,
                 squared: bool = False,
                 p: float = 2.0,

                 # optional structure‐gating parameters (默认不开启)
                 similarity_gate: bool = False,
                 similarity_neg: float = 0.8,
                 similarity_pos: float = 0.2,

                 # feature parameters
                 pre_transform=Gen39AtomFeatures(),

                 # model parameters
                 out_channels: int = 1,
                 convs_layers: list = [64, 128, 256, 512],
                 dense_layers: list = [256, 128, 32],
                 pooling_layer=global_max_pool,
                 batch_norms=torch.nn.BatchNorm1d,
                 dropout_p: float = 0.0,
                 aggregators: list = ['mean', 'min', 'max', 'sum', 'std'],
                 scalers: list = ['identity', 'amplification', 'attenuation'],

                 # other
                 gpuid: int = 0,
                 work_dir: str = './',
                 **model_args):

        super().__init__()
        _fix_reproducibility(42)

        # training control
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # loss params
        self.alpha = alpha
        self.cliff_lower = cliff_lower
        self.cliff_upper = cliff_upper
        self.squared = squared
        self.p = p
        self.similarity_gate = similarity_gate
        self.similarity_neg = similarity_neg
        self.similarity_pos = similarity_pos

        # feature params
        self.pre_transform = pre_transform
        self.in_channels = pre_transform.in_channels
        self.edge_dim = pre_transform.edge_dim

        # model params
        self.out_channels = out_channels
        self.convs_layers = convs_layers
        self.dense_layers = dense_layers
        self.pooling_layer = pooling_layer
        self.batch_norms = batch_norms
        self.dropout_p = dropout_p
        self.aggregators = aggregators
        self.scalers = scalers

        # device & work dir
        self.gpuid = gpuid
        self.work_dir = work_dir
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        torch.cuda.set_device(gpuid)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # arguments passed to ACANet_PNA
        self.model_pub_args = {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'edge_dim': self.edge_dim,
            'convs_layers': self.convs_layers,
            'dense_layers': self.dense_layers,
            'dropout_p': self.dropout_p,
            'pooling_layer': self.pooling_layer,
            'batch_norms': self.batch_norms,
            'aggregators': self.aggregators,
            'scalers': self.scalers,
        }
        self.model_pub_args.update(model_args)
        self.cv_models = []

    def _setup(self, train_dataset, alpha, cliff_lower, cliff_upper):
        """
        Set up ACANet_PNA model, optimizer, ACA loss, and saver.
        """
        deg = get_deg(train_dataset)
        model = ACANet_PNA(**self.model_pub_args, deg=deg).to(self.device)

        aca_loss = ACALoss(alpha=alpha,
                           cliff_lower=cliff_lower,
                           cliff_upper=cliff_upper,
                           squared=self.squared,
                           p=self.p,
                           similarity_gate=self.similarity_gate,
                           similarity_neg=self.similarity_neg,
                           similarity_pos=self.similarity_pos,
                           dev_mode=False)

        saver = SaveBestModel(data_transformer=self.smiles_to_data,
                              save_dir=self.work_dir,
                              save_name='best_model.pth')

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
        return model, optimizer, aca_loss, saver

    def smiles_to_data(self, smiles_list):
        """
        Convert a list of SMILES strings to a list of PyG Data objects
        with node/edge features and fingerprint stored in `data.fp`.
        """
        data_list = []
        for smiles in smiles_list:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            fp = get_morgan_fingerprint(mol)
            data = Data(smiles=smiles, fp=fp)
            data = self.pre_transform(data)
            data_list.append(data)
        return data_list

    def _Xy_to_dataset(self, Xs, y):
        """
        Convert lists Xs (SMILES) and y (labels) into a list of Data objects.
        """
        dataset = []
        for smi, _y in zip(Xs, y):
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smi)
            fp = get_morgan_fingerprint(mol)
            y_tensor = torch.tensor([_y], dtype=torch.float).view(1, -1)
            data = Data(smiles=smi, y=y_tensor, fp=fp)
            data = self.pre_transform(data)
            dataset.append(data)
        return dataset

    def _cv_split(self, y_train, n_folds=5, random_state=42):
        """
        Return a list of dicts with 'inner_train_idx' and 'inner_val_idx' for CV splits.
        """
        y_arr = np.array(y_train).reshape(-1,)
        if len(y_arr) >= 250:
            ss = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
            cutoff = np.median(y_arr)
            labels = [0 if val < cutoff else 1 for val in y_arr]
            splits = [{'inner_train_idx': i, 'inner_val_idx': j}
                      for i, j in ss.split(labels, labels)]
        else:
            idx = np.argsort(y_arr, axis=0).reshape(-1,)
            ts = pd.Series(idx)
            splits = []
            for i in range(n_folds):
                ts_idx = ts.iloc[i::n_folds].tolist()
                tr_idx = ts[~ts.isin(ts_idx)].tolist()
                splits.append({'inner_train_idx': tr_idx, 'inner_val_idx': ts_idx})
        return splits

    def _cv_performance(self,
                        Xs_train,
                        y_train,
                        alpha,
                        cliff_lower,
                        cliff_upper,
                        n_folds=5,
                        total_epochs=300,
                        random_state=42):
        """
        Return a DataFrame of validation RMSE history for each fold.
        """
        splits = self._cv_split(y_train, n_folds=n_folds, random_state=random_state)
        train_dataset = self._Xy_to_dataset(Xs_train, y_train)

        all_histories = []
        for i_split, split in enumerate(splits):
            inner_train = [train_dataset[i] for i in split['inner_train_idx']]
            inner_val = [train_dataset[i] for i in split['inner_val_idx']]

            inner_train_loader = DataLoader(inner_train, batch_size=self.batch_size, shuffle=True)
            inner_val_loader = DataLoader(inner_val, batch_size=self.batch_size)

            model, optimizer, aca_loss, saver = self._setup(
                train_dataset, alpha=alpha, cliff_lower=cliff_lower, cliff_upper=cliff_upper
            )

            history = []
            for epoch in tqdm(range(total_epochs), desc=f'Fold {i_split} Epoch', ascii=True):
                _ = train(inner_train_loader, model, optimizer, aca_loss, self.device)
                val_rmse = test(inner_val_loader, model, self.device)
                history.append(val_rmse)

            ts = pd.Series(history, name=i_split)
            all_histories.append(ts)

        return pd.concat(all_histories, axis=1)

    def opt_cliff_by_trps(self, Xs_train, y_train, iterations=5):
        """
        Optimize cliff_lower/cliff_upper by maximizing number of mined triplets.
        """
        vmin = 0.0
        vmax = 4.1
        cliffs = [round(x, 2) for x in np.arange(vmin, vmax, 0.1)]
        train_dataset = self._Xy_to_dataset(Xs_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        print('Get the best cliff parameter via the number of mined triplets...')
        records = []
        dfs_list = []
        for epoch in tqdm(range(iterations), desc='Epochs', ascii=True):
            for data in train_loader:
                y = data.y.view(-1).to(self.device)
                lower, upper, dfc = get_best_cliff(y, cliffs=cliffs)
                records.append((lower, upper))
                dfs_list.append(dfc)

        df_records = pd.DataFrame(records, columns=['lower', 'upper'])
        best_lower = df_records['lower'].mode().iloc[0]
        best_upper = df_records['upper'].mode().iloc[0]
        self.cliff_lower = best_lower
        self.cliff_upper = best_upper
        print(f'Best cliff_lower/cliff_upper by triplet count: {best_lower}, {best_upper}')

        # average DataFrames across iterations
        dfp = sum(dfs_list) / len(dfs_list)
        plot_dfp_save(dfp, save_dir=self.work_dir)  # 用户需提前定义该函数
        return dfp

    def opt_alpha_by_cv(self,
                        Xs_train,
                        y_train,
                        n_folds=5,
                        n_repeats=1,
                        total_epochs=800,
                        save_rawdata=False,
                        alphas=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
                        random_state=42):
        """
        Optimize alpha via cross‐validation over candidate alphas.
        """
        w = 5
        results = []
        rawdata = []

        for alpha in alphas:
            rmses = []
            for rp in range(n_repeats):
                dfcv = self._cv_performance(
                    Xs_train, y_train,
                    alpha=alpha,
                    cliff_lower=self.cliff_lower,
                    cliff_upper=self.cliff_upper,
                    n_folds=n_folds,
                    total_epochs=total_epochs,
                    random_state=random_state
                )
                rawdata.append({'alpha': alpha, 'repeat': rp, 'dfcv': dfcv})
                rmse_best = dfcv.mean(axis=1).rolling(w).mean().min()
                rmses.append(rmse_best)

            rmse_mean = np.mean(rmses)
            rmse_err = np.std(rmses)
            results.append([alpha, rmse_mean, rmse_err])

        if save_rawdata:
            dump(rawdata, os.path.join(self.work_dir, 'alpha_performance_rawdata.pkl'))

        dfa = pd.DataFrame(results, columns=['alpha', 'rmse', 'rmse_err'])
        best_alpha = dfa.loc[dfa['rmse'].idxmin(), 'alpha']
        self.alpha = best_alpha
        print(f'Best alpha by cross‐validation: {best_alpha}')

        plot_dfa_save(dfa, save_dir=self.work_dir)  # 用户需提前定义该函数
        return dfa

    def opt_cliff_by_cv(self,
                        Xs_train,
                        y_train,
                        n_folds=5,
                        n_repeats=1,
                        total_epochs=800,
                        upper_cliffs=[round(x, 2) for x in np.arange(0.5, 4.5, 0.5)],
                        random_state=42):
        """
        Optimize (cliff_lower, cliff_upper) via nested CV: first find best upper, then best lower.
        """
        w = 3
        results = []

        # 1) find best cliff_upper (fix cliff_lower=0.1)
        for cliff_upper in upper_cliffs:
            cliff_lower = 0.1
            rmses = []
            for rp in range(n_repeats):
                dfcv = self._cv_performance(
                    Xs_train, y_train,
                    alpha=self.alpha,
                    cliff_lower=cliff_lower,
                    cliff_upper=cliff_upper,
                    n_folds=n_folds,
                    total_epochs=total_epochs,
                    random_state=random_state
                )
                rmse_best = dfcv.mean(axis=1).rolling(w).mean().min()
                rmses.append(rmse_best)
            rmses = np.array(rmses)
            results.append([cliff_lower, cliff_upper, rmses.mean(), rmses.std()])

        df1 = pd.DataFrame(results, columns=['cl', 'cu', 'rmse', 'rmse_err'])
        best_upper = df1.loc[df1['rmse'].idxmin(), 'cu']

        # 2) find best cliff_lower (fix cliff_upper=best_upper)
        results = []
        start = 0.1
        stop = best_upper
        cs = np.linspace(start, stop, num=9).round(2)[:-1]  # 8 points
        lower_cliffs = list(cs)
        for cliff_lower in lower_cliffs:
            cliff_upper = best_upper
            rmses = []
            for rp in range(n_repeats):
                dfcv = self._cv_performance(
                    Xs_train, y_train,
                    alpha=self.alpha,
                    cliff_lower=cliff_lower,
                    cliff_upper=cliff_upper,
                    n_folds=n_folds,
                    total_epochs=total_epochs,
                    random_state=random_state
                )
                rmse_best = dfcv.mean(axis=1).rolling(w).mean().min()
                rmses.append(rmse_best)
            rmses = np.array(rmses)
            results.append([cliff_lower, cliff_upper, rmses.mean(), rmses.std()])

        df2 = pd.DataFrame(results, columns=['cl', 'cu', 'rmse', 'rmse_err'])
        dfp = pd.concat([df1, df2]).drop_duplicates(['cl', 'cu']).reset_index(drop=True)
        best = dfp.loc[dfp['rmse'].idxmin()]
        best_lower = best['cl']
        best_upper = best['cu']

        self.cliff_lower = best_lower
        self.cliff_upper = best_upper
        print(f'Best cliff_lower, cliff_upper: {best_lower}, {best_upper}')

        plot_dfc_save(dfp, save_dir=self.work_dir)  # 用户需提前定义该函数
        return dfp

    def opt_epoch_by_cv(self, Xs_train, y_train,
                        n_folds=5,
                        total_epochs=800,
                        random_state=42):
        """
        Optimize number of epochs via cross‐validation.
        """
        dfcv = self._cv_performance(
            Xs_train, y_train,
            alpha=self.alpha,
            cliff_lower=self.cliff_lower,
            cliff_upper=self.cliff_upper,
            n_folds=n_folds,
            total_epochs=total_epochs,
            random_state=random_state
        )
        dfe = dfcv.mean(axis=1).rolling(100).mean()
        best_epoch = dfe.idxmin()
        self.epochs = best_epoch
        print(f'Best epochs by cross‐validation: {best_epoch}')

        plot_dfe_save(dfe, save_dir=self.work_dir)  # 用户需提前定义该函数
        return dfe

    def fit(self,
            Xs_train,
            y_train,
            Xs_val=None,
            y_val=None,
            verbose: int = 1,
            save_model: bool = False):
        """
        Train on (Xs_train, y_train), optionally validate on (Xs_val, y_val).
        """
        train_dataset = self._Xy_to_dataset(Xs_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if Xs_val is None or y_val is None:
            val_loader = None
        else:
            val_dataset = self._Xy_to_dataset(Xs_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        history = []
        model, optimizer, aca_loss, saver = self._setup(
            train_dataset,
            alpha=self.alpha,
            cliff_lower=self.cliff_lower,
            cliff_upper=self.cliff_upper
        )

        for epoch in range(1, self.epochs + 1):
            train_loss = train(train_loader, model, optimizer, aca_loss, self.device)
            if val_loader is None:
                val_rmse = np.nan
                saver(train_loss, epoch, model, optimizer)
            else:
                val_rmse = test(val_loader, model, self.device)
                saver(val_rmse, epoch, model, optimizer)

            if verbose:
                print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f} Test: {val_rmse:.4f}')
            history.append({'Epoch': epoch, 'train_loss': train_loss, 'val_rmse': val_rmse})

        # Load best model and optimizer state
        model.load_state_dict(saver.inMemorySave['model_state_dict'])
        optimizer.load_state_dict(saver.inMemorySave['optimizer_state_dict'])
        model.eval()

        self.model = model
        self.optimizer = optimizer
        self.aca_loss = aca_loss
        self.saver = saver

        # Save configuration
        model_args = {k: str(v) for k, v in self.model.model_args.items()}
        config = {
            'training': {'lr': self.lr, 'epochs': self.epochs, 'batch_size': self.batch_size},
            'loss': {'alpha': self.alpha,
                     'cliff_lower': self.cliff_lower, 'cliff_upper': self.cliff_upper,
                     'squared': self.squared, 'p': self.p,
                     'similarity_gate': self.similarity_gate,
                     'similarity_neg': self.similarity_neg,
                     'similarity_pos': self.similarity_pos},
            'model': model_args
        }
        with open(os.path.join(self.work_dir, 'config.json'), 'w') as fp:
            json.dump(config, fp)

        if save_model:
            self.saver.save()

        return self

    def predict(self, Xs_test):
        """
        Predict on a list of SMILES.
        """
        test_dataset = self.smiles_to_data(Xs_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        test_pred = predict(test_loader, self.model, self.device)
        y_pred = test_pred.cpu().numpy().reshape(-1,)
        return y_pred

    def cv_fit(self, Xs_train, y_train, n_folds=5, verbose=1, random_state=42):
        """
        Fit via cross‐validation, storing each fold’s model in self.cv_models.
        """
        splits = self._cv_split(y_train, n_folds=n_folds, random_state=random_state)
        cv_models = []
        for i, split in enumerate(splits):
            inner_train_x = [Xs_train[idx] for idx in split['inner_train_idx']]
            inner_train_y = [y_train[idx] for idx in split['inner_train_idx']]
            inner_val_x = [Xs_train[idx] for idx in split['inner_val_idx']]
            inner_val_y = [y_train[idx] for idx in split['inner_val_idx']]

            self.fit(inner_train_x, inner_train_y, inner_val_x, inner_val_y, verbose=verbose)
            cv_models.append(self.model)

        self.cv_models = cv_models
        return self

    def cv_predict(self, Xs_test):
        """
        Predict via averaging predictions from each CV fold model.
        """
        assert len(self.cv_models) >= 1, 'Please run cv_fit first!'
        test_dataset = self.smiles_to_data(Xs_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        cv_preds = []
        for model in self.cv_models:
            test_pred = predict(test_loader, model, self.device)
            cv_preds.append(test_pred.cpu().numpy().reshape(-1,))

        y_pred_final = np.mean(cv_preds, axis=0)
        return y_pred_final

    def save(self, mfile: str):
        dump(self, mfile)
        print(f'Saved the model to {mfile}')

    def load(self, mfile: str):
        loaded = load(mfile)
        return loaded

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
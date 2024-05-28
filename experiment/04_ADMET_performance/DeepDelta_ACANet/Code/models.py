# Imports
import os
import tempfile
import shutil
import abc
import pandas as pd
import numpy as np
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold
#import chemprop
from sklearn.ensemble import RandomForestRegressor as RF
#from lightgbm import LGBMRegressor as lgb

# Define abstract class to define interface of models
class abstractDeltaModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, x, y):
        pass

    @abc.abstractmethod
    def predict(self, x):
        pass    




class DeepDelta(abstractDeltaModel):
    epochs = None
    dirpath = None 

    def __init__(self, epochs=5, dirpath = None):
        self.epochs = epochs
        self.dirpath = dirpath


    def fit(self, x, y, metric='r2'):
        
        self.dirpath = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
        # create pairs of training data
        train = pd.merge(x, x, how='cross') 
        y_values = pd.merge(y, y, how='cross')
        train["Y"] = y_values.Y_y - y_values.Y_x
        del y_values 

        temp_datafile = tempfile.NamedTemporaryFile() 
        train.to_csv(temp_datafile.name, index=False)
        
        # store default arguments for ChemProp model
        arguments = [ 
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name, 
            '--dataset_type', 'regression', 
            '--save_dir', self.dirpath,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1', 
            '--epochs', str(self.epochs),
            '--metric', metric,
            '--number_of_molecules', '2',
            '--aggregation', 'sum'
        ]
        
        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x):

        dataset = pd.merge(x, x, how='cross') # Make pairs by cross-merging

        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath,
            '--number_of_molecules', '2'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Predict

        predictions = pd.read_csv(temp_predfile.name)['Y']

        temp_datafile.close()
        temp_predfile.close()

        return predictions
    
    def __str__(self):
        return "DeepDelta" + str(self.epochs)




class Trad_ChemProp(abstractDeltaModel):
    epochs = None
    dirpath = None  
    dirpath_single = None

    def __init__(self, epochs=50, dirpath = None, dirpath_single = None):
        self.epochs = epochs
        self.dirpath = dirpath
        self.dirpath_single = dirpath_single

    def fit(self, x, y, metric='r2'):
        self.dirpath_single = tempfile.NamedTemporaryFile().name # use temporary file to store model
        
        train = pd.DataFrame(np.transpose(np.vstack([x,y])),columns=["X","Y"])

        temp_datafile = tempfile.NamedTemporaryFile()
        train.to_csv(temp_datafile.name, index=False)

        # store default arguments for ChemProp model
        arguments = [
            '--data_path', temp_datafile.name,
            '--separate_val_path', temp_datafile.name,
            '--dataset_type', 'regression',
            '--save_dir', self.dirpath_single,
            '--num_folds', '1',
            '--split_sizes', '1.0', '0', '0',
            '--ensemble_size', '1',
            '--epochs', str(self.epochs),
            '--number_of_molecules', '1',
            '--metric', metric, 
            '--aggregation', 'sum'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) # Train

        temp_datafile.close()


    def predict(self, x): 

        dataset = pd.DataFrame(x)
        temp_datafile = tempfile.NamedTemporaryFile()
        dataset.to_csv(temp_datafile.name, index=False)
        temp_predfile = tempfile.NamedTemporaryFile()

        arguments = [
            '--test_path', temp_datafile.name,
            '--preds_path', temp_predfile.name, 
            '--checkpoint_dir', self.dirpath_single,
            '--number_of_molecules', '1'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        chemprop.train.make_predictions(args=args) # Make prediction

        predictions = pd.read_csv(temp_predfile.name)['Y'] 

        preds = pd.merge(predictions,predictions,how='cross') # Cross merge to make pairs

        temp_datafile.close()
        temp_predfile.close()

        return preds.Y_y - preds.Y_x   # Calculate and return the delta values
    
    def __str__(self):
        return "ChemProp" + str(self.epochs)




class Trad_RF(abstractDeltaModel):
    model = None

    def __init__(self):
        self.model = RF()

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        self.model.fit(fps,y) # Fit using traditional methods

    def predict(self, x):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        predictions = pd.DataFrame(self.model.predict(fps)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x']  # Calculate and return the delta values
    
    def __str__(self):
        return "RandomForest"




class Trad_LGBM(abstractDeltaModel):
    model = None

    def __init__(self, subsample=0.1, subsample_freq = 1, min_child_samples = 20): # Default min_child_samples is 20
        self.model = lgb(subsample = subsample, subsample_freq = subsample_freq, min_child_samples = min_child_samples)

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        self.model.fit(fps,y) # Fit using traditional methods

    def predict(self, x):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        
        predictions = pd.DataFrame(self.model.predict(fps)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x'] # Calculate and return the delta values
    
    def __str__(self):
        return "LGBM"
        
        

class Delta_LGBM(abstractDeltaModel):
    model = None

    def __init__(self, subsample=0.1, subsample_freq = 1):
        self.model = lgb(subsample=subsample, subsample_freq=subsample_freq)  

    def fit(self, x, y, metric='r2'):
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool)), 'Y':  y})
        pair_data = pd.merge(data, data, how='cross') # Cross merge data to make pairs
        
        pair_data["Fingerprint"] =  pair_data.Fingerprint_x.combine(pair_data.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
        pair_data["Delta"] = pair_data.Y_y - pair_data.Y_x # calculate Delta values
        
        pair_data.drop(['Fingerprint_x','Fingerprint_y','Y_x','Y_y'], axis=1, inplace=True) # remove unnecessary data
        self.model.fit(np.vstack(pair_data.Fingerprint.to_numpy()),pair_data.Delta) # fit model on pair training data
        del pair_data # remove unnecessary data

    def predict(self, x):
        predictions = []
        mols = [Chem.MolFromSmiles(s) for s in x]
        fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
        data = pd.DataFrame(data={'Fingerprint': list(np.array(fps).astype(bool))})
        pair_data = pd.merge(data, data, how='cross') # Cross merge data to make pairs
        pair_data["Fingerprint"] = pair_data.Fingerprint_x.combine(pair_data.Fingerprint_y, np.append) # concatenate ExplicitBitVec objects from RDKIT
        pair_data.drop(['Fingerprint_x','Fingerprint_y'], axis=1, inplace=True) # remove unnecessary data
        
        predictions += [self.model.predict(np.vstack(pair_data.Fingerprint.to_numpy()))] # make predictions
        return np.concatenate(predictions)
    
    def __str__(self):
        return "DeltaLGBM"



import sys
sys.path.insert(0, '/home/shenwanxiang/Research/bidd-clsar')
from clsar import ACANet
class ACANetOOTB(abstractDeltaModel):

    def __init__(self, dirpath = './', **kwargs):
        self.dirpath = dirpath
        self.model = ACANet(work_dir = dirpath, **kwargs)

    def fit(self, x, y, metric='r2'):

        #print(x)
        
        x = x.values
        y = y.values

        ## get loss hyperparameters (cliff_lower, cliff_upper, and alpha) by training set
        dfp = self.model.opt_cliff_by_trps(x, y, iterations = 5)
        #dfp = self.model.opt_cliff_by_cv(x, y, total_epochs=50, n_repeats=1)
        #dfa = self.model.opt_alpha_by_cv(x, y, total_epochs=100, n_repeats=1)
        #opt_cliff_by_trps
        
        ## fit model
        self.model.cv_fit(x, y, verbose=1)
        

    def predict(self, x):
        x = x.values
        predictions = pd.DataFrame(self.model.cv_predict(x)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x']  # Calculate and return the delta values

    
    def __str__(self):
        return "ACANetOOTB"


class ACANetOPTPoC(abstractDeltaModel):

    def __init__(self, dirpath = './', **kwargs):
        self.dirpath = dirpath
        self.model = ACANet(work_dir = dirpath, **kwargs)


    def fit(self, x, y, metric='r2'):

        #print(x)
        
        x = x.values
        y = y.values

        dfp = self.model.opt_cliff_by_cv(x, y, total_epochs=100, n_repeats=5)
        dfa = self.model.opt_alpha_by_cv(x, y, total_epochs=100, n_repeats=5)
        ## fit model
        self.model.cv_fit(x, y, verbose=1)
        

    def predict(self, x):
        x = x.values
        predictions = pd.DataFrame(self.model.cv_predict(x)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x']  # Calculate and return the delta values

    
    def __str__(self):
        return "ACANetOPTPoC" 
    
class ACANetOPT(abstractDeltaModel):

    def __init__(self, dirpath = './', **kwargs):
        self.dirpath = dirpath
        self.model = ACANet(work_dir = dirpath, **kwargs)


    def fit(self, x, y, metric='r2'):

        #print(x)
        
        x = x.values
        y = y.values

        dfp = self.model.opt_cliff_by_cv(x, y, total_epochs=50, n_repeats=1)
        dfa = self.model.opt_alpha_by_cv(x, y, total_epochs=100, n_repeats=1)
        ## fit model
        self.model.cv_fit(x, y, verbose=1)
        

    def predict(self, x):
        x = x.values
        predictions = pd.DataFrame(self.model.cv_predict(x)) # Predict using traditional methods
        results = pd.merge(predictions,predictions,how='cross') # Cross merge into pairs after predictions
        return results['0_y'] - results['0_x']  # Calculate and return the delta values

    
    def __str__(self):
        return "ACANetOPT" 
    
    








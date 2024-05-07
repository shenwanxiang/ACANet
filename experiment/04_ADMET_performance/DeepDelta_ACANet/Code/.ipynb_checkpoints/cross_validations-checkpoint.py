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
import chemprop
from sklearn.ensemble import RandomForestRegressor as RF
from lightgbm import LGBMRegressor as lgb
from models import *

def cross_validation(x, y, prop, model, k=10, seed=1): # provide option to cross validate with x and y instead of file
  kf = KFold(n_splits=k, random_state=seed, shuffle=True)
  cnt = 1 # Used to keep track of current fold
  preds = []
  vals  = []

  for train, test in kf.split(x):
        model.fit(x[train],y[train]) # Fit on training data
        preds = np.append(preds, model.predict(x[test])) # Predict on testing data
        y_pairs = pd.merge(y[test],y[test],how='cross') # Cross-merge data values
        vals = np.append(vals, y_pairs.Y_y - y_pairs.Y_x) # Calculate true delta values
        
        if seed == 1: # Saving individual folds for mathematical invariants analysis
            results = [preds]
            pd.DataFrame(results).to_csv("{}_{}_Individual_Fold_{}.csv".format(prop, model, cnt), index=False)
            # If you .T the dataframe, then the first column is predictions
            cnt +=1
    
  return [vals,preds] # Return true delta values and predicted delta values


def cross_validation_file(data_path, prop, model, k=10, seed=1): # Cross-validate from a file
  df = pd.read_csv(data_path)
  x = df[df.columns[0]]
  y = df[df.columns[1]]
  return cross_validation(x,y,prop,model,k,seed)



###################
####  5x10 CV  ####
###################

properties = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']
models = [DeepDelta(), Trad_ChemProp(), Trad_RF(), Trad_LGBM(), Delta_LGBM()]

for model in models:
    for prop in properties:
        delta = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE']) # For storing results
        for i in range(5): # Allow for 5x10-fold cross validation
            dataset = '../Datasets/Benchmarks/{}.csv'.format(prop) # Training dataset
            
            if str(model) == 'LGBM' and prop == 'FUBrain': # Use min_child_sample of 5 for smallest dataset
              results = cross_validation_file(data_path=dataset, prop = prop, model=Trad_LGBM(min_child_samples = 5), k=10, seed = i) # Run cross-validation
            else: # Otherwise use standard procedures
              results = cross_validation_file(data_path=dataset, prop = prop, model=model, k=10, seed = i) # Run cross-validation

            pd.DataFrame(results).to_csv("{}_{}_{}.csv".format(prop, str(model), i), index=False) # Save results
            # If you .T the dataframe, then the first column is ground truth, the second is predictions

            # Read saved dataframe to calculate statistics
            df = pd.read_csv("{}_{}_{}.csv".format(prop, model, i)).T
            df.columns =['True', 'Delta']
            trues = df['True'].tolist()
            preds = df['Delta'].tolist() 
            
            # Calculate statistics for each round
            pearson = stats.pearsonr(trues, preds)
            MAE = metrics.mean_absolute_error(trues, preds)
            RMSE = math.sqrt(metrics.mean_squared_error(trues, preds))
            scoring = pd.DataFrame({'Pearson\'s r': [round(pearson[0], 4)], 'MAE': [round(MAE, 4)], 'RMSE': [round(RMSE, 4)]})
            delta = pd.concat([delta, scoring])

        # Calculate overall statistics 
        average = pd.DataFrame({'Pearson\'s r': [round(np.mean(delta['Pearson\'s r']), 3)], 'MAE': [round(np.mean(delta['MAE']), 3)], 'RMSE': [round(np.mean(delta['RMSE']), 3)]})
        std = pd.DataFrame({'Pearson\'s r': [round(np.std(delta['Pearson\'s r']), 3)], 'MAE': [round(np.std(delta['MAE']), 3)], 'RMSE': [round(np.std(delta['RMSE']), 3)]})
        delta = pd.concat([delta, average])
        delta = pd.concat([delta, std])
        delta = delta.set_index([pd.Index([1, 2, 3, 4, 5, 'Avg', 'Std. Dev.'])])
        delta.to_csv("{}_{}_delta_scoring.csv".format(prop, model)) # Save data
        



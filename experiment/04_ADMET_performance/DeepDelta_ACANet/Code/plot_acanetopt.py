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
# import chemprop
from sklearn.ensemble import RandomForestRegressor as RF
#from lightgbm import LGBMRegressor as lgb
from models import ACANetOPTPoC, ACANetOPT 

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
            #pd.DataFrame(results).to_csv("../Results/ACANetOPT/{}_{}_Individual_Fold_{}.csv".format(prop, model, cnt), index=False)
            # If you .T the dataframe, then the first column is predictions
            cnt +=1

    return [vals, preds] # Return true delta values and predicted delta values


def cross_validation_file(data_path, prop, model, k=10, seed=1): # Cross-validate from a file
    df = pd.read_csv(data_path)
    x = df[df.columns[0]]
    y = df[df.columns[1]]
    return cross_validation(x,y,prop,model,k,seed)



###################
####  5x10 CV  ####
###################

properties = ['HalfLife']
for prop in properties:
    dirpath = os.path.join('./plot/acanetopt_poc', prop)
    
    models = [ACANetOPTPoC(dirpath=dirpath, gpuid=1)]
    for model in models:
        delta = pd.DataFrame(columns=['Pearson\'s r', 'MAE', 'RMSE']) # For storing results
        for i in range(5): # Allow for 5x10-fold cross validation
            if i== 0:
                dataset = '../Datasets/Benchmarks/{}.csv'.format(prop) # Training dataset
                results = cross_validation_file(data_path=dataset, prop = prop, model=model, k=10, seed = i) 
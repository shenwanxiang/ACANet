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

properties = ['Sol', 'VDss'] 
models = [DeepDelta(), Trad_ChemProp(), Trad_RF(), Trad_LGBM(), Delta_LGBM()]

for model in models:
    for prop in properties:
        dataset = '../Datasets/Benchmarks/{}.csv'.format(prop) # For training
        pred_dataset = '../Datasets/External/{}ExternalTestSetFiltered.csv'.format(prop) # For prediction
        
        # Fit model on entire training dataset
        df = pd.read_csv(dataset)
        x = df[df.columns[0]]
        y = df[df.columns[1]]
        model.fit(x,y) 
        
        # Predict on cross-merged external test set
        df = pd.read_csv(pred_dataset)
        x = df[df.columns[0]]
        y = df[df.columns[1]]
        y_pairs = pd.merge(y, y, how='cross') # Cross-merge into pairs
        vals = y_pairs.Y_y - y_pairs.Y_x # Calculate delta values
        preds = model.predict(x) # Make predictions
        
        results = [vals, preds] # Save the true delta values and predictions
        pd.DataFrame(results).to_csv("{}_{}_ExtTestSet.csv".format(prop, model), index=False) # Save results
        #If you .T the dataframe, then the first column is ground truth, the second is predictions


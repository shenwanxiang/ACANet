# Imports
import pandas as pd
import numpy as np
import math


datasets = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']
scoring = pd.DataFrame(columns=['Dataset', 'MAE1', 'RMSE1', 'MAE2', 'RMSE2', 'COR2', 'MAE3', 'RMSE3']) # For results

def MAE(x, n): # Calculate Mean Absolute Error (MAE)
  return np.sum(np.abs(x)) / n

def RMSE(x, n): # Calculate Root Mean Squared Error (RMSE)
  return np.sqrt(np.sum(np.square(x)) / n)


for dataset in datasets:

  n_diagonal = 0
  n_reverse = 0
  n_triangle_paired = 0
  diagonal = []
  reverse_paired = []
  triangle_paired = []


  for cnt in range(1,11): # Go through each fold of cross-validation individually
    preds = pd.read_csv('../Results/DeepDelta_Individual_CV_Folds/{}_DeepDelta_Individual_Fold_{}.csv'.format(dataset, cnt)).T

    n_fold = int(np.sqrt(len(preds))) # Store the number of molecule pairs in each fold
    matrix = np.reshape(np.array(preds), (n_fold, n_fold)) # Create a matrix from the molecule pairs

    diagonal = np.append(np.array(diagonal), np.diag(matrix)) # For Eq1, include the same molecule pairs
    n_diagonal += n_fold # For Eq1

    reverse_paired = np.append(np.array(reverse_paired), (matrix + np.transpose(matrix)).flatten()) # For Eq2, include the reversed predictions
    n_reverse += n_fold*n_fold # For Eq2

    triangle_paired_fold = []
    for i in range(n_fold):
      for ii in range(n_fold):
        triangle_paired_fold.extend(matrix[i,:] + matrix[:,ii] - matrix[i,ii]) # For Eq3, calculate all triangle groupings

    triangle_paired =  np.append(np.array(triangle_paired), triangle_paired_fold) # For Eq3, include the triangle groupings
    n_triangle_paired += n_fold**3 # For Eq3

  ####################
  # Eq1: DD(x,x) = 0 #
  #################### 

  MAE1 = MAE(diagonal, n_diagonal)
  RMSE1 = RMSE(diagonal, n_diagonal)
  
  ####################


  ###########################
  # Eq2: DD(x,y) = -DD(y,x) #
  ###########################

  MAE2 = MAE(reverse_paired, n_reverse)
  RMSE2 = RMSE(reverse_paired, n_reverse)
  COR2 = np.corrcoef(matrix.flatten(), np.transpose(matrix).flatten())[0,1]

  ###########################


  ####################################
  # Eq3: DD(x,y) + DD(y,z) = DD(x,z) #
  ####################################
    
  MAE3 = MAE(triangle_paired, (n_triangle_paired))
  RMSE3 = RMSE(triangle_paired, (n_triangle_paired))
    
  ####################################

  scores = pd.DataFrame({'Dataset': [dataset], 'MAE1': [round(MAE1, 4)], 'RMSE1': [round(RMSE1, 4)], 
                    'MAE2': [round(MAE2, 4)], 'RMSE2': [round(RMSE2, 4)], 'COR2': [round(COR2, 4)],
                    'MAE3': [round(MAE3, 4)], 'RMSE3': [round(RMSE3, 4)]})
  scoring = pd.concat([scoring, scores]) # Add results from each fold

scoring.to_csv("Mathematical_Invariants_Results.csv", index = False) # Save results








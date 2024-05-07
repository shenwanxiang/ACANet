# Imports
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold 
import math
from sklearn import metrics
from scipy import stats as stats
from sklearn.model_selection import KFold


#########################
### Scaffold Analysis ###
#########################

# Read local training data
datasets = ['FUBrain', 'RenClear', 'FreeSolv', 'MicroClear', 'HemoTox', 'HepClear', 'Caco2', 'Sol', 'VDss', 'HalfLife']

all_scoresNM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs of Nonmatching Scaffolds
all_scoresM = pd.DataFrame(columns=['Dataset', 'Pearson\'s r RF', 'MAE RF', 'RMSE RF',
                                    'Pearson\'s r CP', 'MAE CP', 'RMSE CP', 'Pearson\'s r DD', 'MAE DD', 'RMSE DD']) # For pairs with Matching Scaffolds

# Evaluate Matching and Non-matching Scaffold Pairs for all Datasets
for name in datasets:
    dataframe = pd.read_csv("../Datasets/Benchmarks/{}.csv".format(name))

    # 3 Models to Evaluate
    # 1 - Random Forest 
    predictions_RF = pd.read_csv('../Results/RandomForest/{}_RandomForest_1.csv'.format(name)).T
    predictions_RF.columns =['True', 'Delta']
    # 2 - ChemProp
    predictions_CP = pd.read_csv('../Results/ChemProp50/{}_ChemProp50_1.csv'.format(name)).T
    predictions_CP.columns =['True', 'Delta']
    # 3 - DeepDelta
    predictions_DD = pd.read_csv('../Results/DeepDelta5/{}_DeepDelta5_1.csv'.format(name)).T
    predictions_DD.columns =['True', 'Delta']

    # Prepare Scaffolds
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in mols]
    data = pd.DataFrame(data={'Scaffold':  scaffolds})
    del dataframe

    # Emulate previous train-test splits and save the scaffolds from this
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    datapoint_x = []
    datapoint_y = []

    for train_index, test_index in cv.split(data):
        train_df = data[data.index.isin(train_index)]
        test_df = data[data.index.isin(test_index)]
        pair_subset_test = pd.merge(test_df, test_df, how='cross')
        datapoint_x += [pair_subset_test.Scaffold_x]
        datapoint_y += [pair_subset_test.Scaffold_y]
        del pair_subset_test

    datapoints = pd.DataFrame(data={'X':  np.concatenate(datapoint_x), 'Y':  np.concatenate(datapoint_y)})

    # Add the actual deltas and predicted deltas
    trues = predictions_CP['True']
    trues = [float(i) for i in trues]
    datapoints['True'] = trues

    DeltasRF = predictions_RF['Delta']
    DeltasRF = [float(i) for i in DeltasRF]
    datapoints['DeltaRF'] = DeltasRF

    DeltasCP = predictions_CP['Delta']
    DeltasCP = [float(i) for i in DeltasCP]
    datapoints['DeltaCP'] = DeltasCP

    DeltasDD = predictions_DD['Delta']
    DeltasDD = [float(i) for i in DeltasDD]
    datapoints['DeltaDD'] = DeltasDD

    # Grab the datapoints with matching scaffolds into a separate dataframe
    final_df = pd.DataFrame(columns = ['X', 'Y', 'True', 'DeltaRF', 'DeltaCP', 'DeltaDD'])
    for i in range(len(datapoints)):
        if datapoints['X'][i] == datapoints['Y'][i]:
            inter_df = pd.DataFrame({'X': [datapoints['X'][i]], 'Y': [datapoints['Y'][i]], 'True': [datapoints['True'][i]], 'DeltaRF': [datapoints['DeltaRF'][i]], 'DeltaCP': [datapoints['DeltaCP'][i]], 'DeltaDD': [datapoints['DeltaDD'][i]]})
            final_df = pd.concat([final_df, inter_df])

    # Grab the nonmatching datapoints
    nonmatching = pd.merge(datapoints,final_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

    # Run Stats - Non-matching Scaffolds
    pearson_NM_RF = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaRF']))
    MAE_NM_RF = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaRF']))
    RMSE_NM_RF = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaRF'])))

    pearson_NM_CP = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaCP']))
    MAE_NM_CP = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaCP']))
    RMSE_NM_CP = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaCP'])))

    pearson_NM_DD = stats.pearsonr(nonmatching["True"], (nonmatching['DeltaDD']))
    MAE_NM_DD = metrics.mean_absolute_error(nonmatching["True"], (nonmatching['DeltaDD']))
    RMSE_NM_DD = math.sqrt(metrics.mean_squared_error(nonmatching["True"], (nonmatching['DeltaDD'])))

    scoringNM = pd.DataFrame({'Dataset': [name], 'Pearson\'s r RF': [round(pearson_NM_RF[0], 4)], 'MAE RF': [round(MAE_NM_RF, 4)], 'RMSE RF': [round(RMSE_NM_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_NM_CP[0], 4)], 'MAE CP': [round(MAE_NM_CP, 4)], 'RMSE CP': [round(RMSE_NM_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_NM_DD[0], 4)], 'MAE DD': [round(MAE_NM_DD, 4)], 'RMSE DD': [round(RMSE_NM_DD, 4)]})

    all_scoresNM = pd.concat([all_scoresNM, scoringNM])

    # Run Stats - Matching Scaffolds
    pearson_M_RF = stats.pearsonr(final_df["True"], (final_df['DeltaRF']))
    MAE_M_RF = metrics.mean_absolute_error(final_df["True"], (final_df['DeltaRF']))
    RMSE_M_RF = math.sqrt(metrics.mean_squared_error(final_df["True"], (final_df['DeltaRF'])))

    pearson_M_CP = stats.pearsonr(final_df["True"], (final_df['DeltaCP']))
    MAE_M_CP = metrics.mean_absolute_error(final_df["True"], (final_df['DeltaCP']))
    RMSE_M_CP = math.sqrt(metrics.mean_squared_error(final_df["True"], (final_df['DeltaCP'])))

    pearson_M_DD = stats.pearsonr(final_df["True"], (final_df['DeltaDD']))
    MAE_M_DD = metrics.mean_absolute_error(final_df["True"], (final_df['DeltaDD']))
    RMSE_M_DD = math.sqrt(metrics.mean_squared_error(final_df["True"], (final_df['DeltaDD'])))

    scoringM = pd.DataFrame({'Dataset': [name], 'Pearson\'s r RF': [round(pearson_M_RF[0], 4)], 'MAE RF': [round(MAE_M_RF, 4)], 'RMSE RF': [round(RMSE_M_RF, 4)],
                            'Pearson\'s r CP': [round(pearson_M_CP[0], 4)], 'MAE CP': [round(MAE_M_CP, 4)], 'RMSE CP': [round(RMSE_M_CP, 4)],
                            'Pearson\'s r DD': [round(pearson_M_DD[0], 4)], 'MAE DD': [round(MAE_M_DD, 4)], 'RMSE DD': [round(RMSE_M_DD, 4)]})

    all_scoresM = pd.concat([all_scoresM, scoringM])

all_scoresNM.to_csv("DeepDelta_Scaffold_NonMatching.csv", index = False) # Save results for unmatched scaffold pairs
all_scoresM.to_csv("DeepDelta_Scaffold_Matching.csv", index = False) # Save results for matched scaffold pairs

#########################




###########################
### Similarity Analysis ###
###########################

for name in datasets:
    dataframe = pd.read_csv("../Datasets/Benchmarks/{}.csv".format(name))

    # Prepare fingerprints
    mols = [Chem.MolFromSmiles(s) for s in dataframe.SMILES]
    fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m,2)) for m in mols]
    dataframe["Fingerprint"] = fps
    data = pd.DataFrame(data={'FP':  dataframe.Fingerprint.to_numpy()})
    del dataframe

    # Set up for cross validation
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=10, random_state=1, shuffle=True)

    # Perform cross validation and save fingerprints
    FA = [] # For the first fingerprint
    FB = [] # For the second fingerprint
    for train_index, test_index in cv.split(data):
        train_df = data[data.index.isin(train_index)]
        test_df = data[data.index.isin(test_index)]
        pair_subset_test = pd.merge(test_df, test_df, how='cross')
        FA += [pair_subset_test.FP_x]
        FB += [pair_subset_test.FP_y]
        del pair_subset_test

    # Calculate the similarity values
    similarity_list = []
    for i in range(len(np.concatenate(FA))):
        similarity_list.append(DataStructs.TanimotoSimilarity(DataStructs.cDataStructs.CreateFromBitString("".join(np.concatenate(FA)[i].astype(str))), DataStructs.cDataStructs.CreateFromBitString("".join(np.concatenate(FB)[i].astype(str)))))

    # Export the csv containing similarity values
    dataframe = pd.DataFrame(similarity_list)
    dataframe.rename(columns={0: 'Tanimoto'}, inplace = True)
    dataframe.to_csv('{}_CV_Similarity_Scores.csv'.format(name), index = False)
    
###########################
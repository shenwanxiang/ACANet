import pandas as pd
import numpy as np
import os

from rdkit import Chem
import networkx as nx
from util import TestbedDataset 




def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    #print(smile, type(smile))
    mol = Chem.MolFromSmiles(smile)#从smiles得到分子
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index


datasets = ['SSD']
for dataset in datasets:
    print('convert data to graph', dataset)
    fpath = 'data/' + dataset + '/'


# 根据smile得到图节点边特征，存到simile_graph字典中
compound_iso_smiles = []
for dt_name in ['SSD']:
    opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv('C:\\Users\CC\Desktop\ceshi\data\SSD_train.csv')
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        #df = pd.read_csv('C:/Users/CC/Desktop/ceshi/data/123.csv')
        #alltargets_dict = dict(zip(alltargets_list, range(len(alltargets_list))))
        compound_iso_smiles += list( df['Smiles'] )

compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smi in compound_iso_smiles:
    if type(smi) != float: #有时候会是float
        g = smile_to_graph(smi)
        smile_graph[smi] = g


# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    #if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))): #！！
        df_train = pd.read_csv('data/' + dataset + '_train.csv')
        df_test = pd.read_csv('data/' + dataset + '_test.csv')
        #print(len(list(df['smiles'])),len(list(df['target_num'])))
        train_drugs, train_Y = list(df_train['Smiles']), list(df_train['label'])
        test_drugs, test_Y = list(df_test['Smiles']), list(df_test['label'])
        data_list = []
        train_drugs, train_Y = np.asarray(train_drugs), np.asarray(train_Y)
        test_drugs, test_Y = np.asarray(test_drugs), np.asarray(test_Y)
        # make data PyTorch Geometric ready
        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, t=train_Y, smile_graph= smile_graph)
        test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, t=test_Y, smile_graph= smile_graph)
        #print('preparing ', dataset + '_test.pt in pytorch format!')
        #print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        #print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')
         print('are already created')

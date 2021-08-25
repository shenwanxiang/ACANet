import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])#共78维
def edge_features(edge):
    #print(edge.GetBondType())
    #print(edge.GetBondDir())
    return np.array(one_of_k_encoding(edge.GetBondType(), [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]) +
                    one_of_k_encoding(edge.GetBondDir(), [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT])
                   )#共7维
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
'''
提取部分特征
'''
def mol_to_graph_data_obj_onehot(mol):
    #print(smile, type(smile))
    #mol = Chem.MolFromSmiles(smile)#从smiles得到分子
    c_size = mol.GetNumAtoms()
    atom_attr = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        #@@@ 这里除了一下 @@@
        atom_attr.append( feature / sum(feature) )
    x = torch.tensor(np.array(atom_attr), dtype=torch.float64)

    edge_attr = []
    edge_list = []
    for bond in mol.GetBonds():
        edge_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_list.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        feature = edge_features(bond)
        edge_attr.append(feature)
        edge_attr.append(feature)
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr))

    #g = nx.Graph(edges).to_directed() #@@@ 为啥要再变成nx
    #edge_index = []
    #for e1, e2 in g.edges:
    #    edge_index.append([e1, e2])
    #data = Data(x=features, edge_index = edge_index, edge_attribute = )
    #return c_size, features, edge_index

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr )
    return data

'''
增加至115个原子特征， 数据scale
'''
from atomfeat import Featurizer
SMI = 'BC[C@H](I)C[C@@H](CBr)C1=C(C#N)C(=CC([NH2+]C(=O)[C@H](C)S)=C1Cl)P(=O)=O'
df = pd.DataFrame(Featurizer(SMI).atomfeats)
df1 = df[df.columns[:76]] #binary
df2 = df[df.columns[76:]] #numeric
x2_max = df2.max().values
x2_min = df2.min().values
def scale_continuous_feat(x2):
    return (x2_max-x2-0.001) / (x2_max-x2_min)

def mol_to_graph_data_obj_custom(mol, scale = False):
    ## 115 dim atom feats
    atomfeats = Featurizer(Chem.MolToSmiles(mol)).atomfeats
    df = pd.DataFrame(atomfeats)
    df1 = df[df.columns[:76]] #binary
    df2 = df[df.columns[76:]] #numeric
    x1 = df1.values
    x2 = df2.values
    if scale:
        x2 = scale_continuous_feat(x2)
    atom_attr  = np.concatenate([x1, x2], axis=1)
    x = torch.tensor(atom_attr, dtype=torch.float64)
    
    edge_attr = []
    edge_list = []
    for bond in mol.GetBonds():
        edge_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_list.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        feature = edge_features(bond)
        edge_attr.append(feature)
        edge_attr.append(feature)
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr))

    #g = nx.Graph(edges).to_directed() #@@@ 为啥要再变成nx
    #edge_index = []
    #for e1, e2 in g.edges:
    #    edge_index.append([e1, e2])
    #data = Data(x=features, edge_index = edge_index, edge_attribute = )
    #return c_size, features, edge_index
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr )
    return data



def mol_to_graph_data_obj_pseudo(mol, dim = 115, seed = 123):
    '''
    dim: 原子特征的维度， 该方法生成随机的 “伪特征”
    '''
    torch.manual_seed(seed)
    ## gennerate a pseudo feature lookup table, 1000 atoms to lookup
    emb = torch.nn.Embedding(1000, dim)
    torch.nn.init.xavier_uniform_(emb.weight.data)
    pseudo_feat_lookup_array = emb.weight.data.numpy()
    pseudo_atom_feats = []
    for atom in mol.GetAtoms():
        aid = atom.GetAtomicNum()
        pseudo_atom_feats.append(pseudo_feat_lookup_array[aid])
    x = torch.tensor(np.stack(pseudo_atom_feats, axis=0), dtype=torch.float64)
    
    edge_attr = []
    edge_list = []
    for bond in mol.GetBonds():
        edge_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_list.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        feature = edge_features(bond)
        edge_attr.append(feature)
        edge_attr.append(feature)
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr))

    #g = nx.Graph(edges).to_directed() #@@@ 为啥要再变成nx
    #edge_index = []
    #for e1, e2 in g.edges:
    #    edge_index.append([e1, e2])
    #data = Data(x=features, edge_index = edge_index, edge_attribute = )
    #return c_size, features, edge_index
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr )
    return data




def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # 这里是有指定原子的一些特征比如atom type,  chirality tag，这是两维的，每个维度一个值表示类型，比如【3，5】
    atom_features_list = [] #两维列表
    for atom in mol.GetAtoms(): #对每个原子提取特征
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # 边的一些特征比如bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr) #转化为torch_geometric的graoh类型

    return data
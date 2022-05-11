import os
import os.path as osp
import re
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn.models import AttentiveFP

from atomfeat import Featurizer


#def scale_continuous_feat(x2):
#    return (x2_max - x2 - 0.001) / (x2_max - x2_min)
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])#共78维
def edge_features(edge):
    print(edge.GetBondType())
    print(edge.GetBondDir())
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



def data_to_custom(data, scale=True):
    ## 115 dim atom feats
    #atomfeats = Featurizer(Chem.MolToSmiles(mol)).atomfeats
    print(3333)
    atomfeats = Featurizer(data.smiles).atomfeats
    print(atomfeats)
    df = pd.DataFrame(atomfeats)
    df1 = df[df.columns[:76]].fillna(0)  # binary
    df2 = df[df.columns[76:]].fillna(0)  # numeric
    x2_max = df2.max().values
    x2_min = df2.min().values
    x1 = df1.values
    x2 = df2.values
    print('MAX:',x2_max)
    print('MIN',x2_min)
    x1 = x1/np.sum(x1)
    
    print('x1:',x1[0])
    print('x2:',x2[0])
    #exit()
    if scale:
        #x2 = scale_continuous_feat(x2)
        x2 = (x2_max - x2 - 0.0001) / (x2_max - x2_min + 0.001)
        
        print('scalex2:',x2)
    
    atom_attr = np.concatenate([x1, x2], axis=1)
    x = torch.tensor(atom_attr, dtype=torch.float64)
    data.x = x
    
    edge_attr = []
    edge_list = []
    
    mol = Chem.MolFromSmiles(data.smiles)
    for bond in mol.GetBonds():
        edge_list.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_list.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))
        feature = edge_features(bond)
        edge_attr.append(feature)
        edge_attr.append(feature)
    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr))
    
    data.edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    data.edge_attr = edge_attr

    return data


x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}






class GenAttentiveFeatures(object):
    '''
    AttentiveFP 39 node features generation
    '''
    def __init__(self):
        self.symbols = [
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
            'Te', 'I', 'At', 'other'
        ]

        self.hybridizations = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

    def __call__(self, data):
        # Generate AttentiveFP features according to Table 1.
        mol = Chem.MolFromSmiles(data.smiles)

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.
            degree = [0.] * 6
            degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(
                atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.

            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type)
            xs.append(x)

        data.x = torch.stack(xs, dim=0)

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[self.stereos.index(bond.GetStereo())] = 1.

            edge_attr = torch.tensor(
                [single, double, triple, aromatic, conjugation, ring] + stereo)

            edge_attrs += [edge_attr, edge_attr]

        if len(edge_attrs) == 0:
            data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            data.edge_index = torch.tensor(edge_indices).t().contiguous()
            data.edge_attr = torch.stack(edge_attrs, dim=0)

        return data


class GenAtomFeatures(object):
    '''
    Our features: @todo
    '''
    #from atomfeat import Featurizer
    def __init__(self,feature_type):
        #super(GenAtomFeatures, self).__init__()
        self.feature_type = feature_type
    def __call__(self, data): 
        #print(3333)
        
        if self.feature_type == 'custom':
            #mol = Chem.MolFromSmiles(data.smiles)
            print(data)
            data = data_to_custom(data, scale=True)
            return data


class LSSInhibitor(InMemoryDataset):
    r"""The benchmark datasets of low-sample size narrow scaffold inhibitors, 
    containing inhibitors from differnt drug targets of GPCRs, Kinases, Nuclear Receptors, Proteases, Transporters, and Other Enzymes.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"mGluR2"`,
            :obj:`"USP7"`, :obj:`"MTH1"`, :obj:`"RIP2"`, :obj:`"PKCi"`,
            :obj:`"PHGDH"`, :obj:`"RORg"`, :obj:`"IDO1"`, :obj:`"KLK5"`,
            :obj:`"Notum"`, :obj:`"EAAT3"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    
    url = 'https://raw.githubusercontent.com/bidd-group/LSSinhibitors/main/data/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]

    names = {
        'mglur2': ['mGluR2', 'mGluR2.csv', 'mGluR2', 7, 12],
        'usp7': ['USP7', 'USP7.csv', 'USP7', 7, 12],
        'mth1': ['MTH1', 'MTH1.csv', 'MTH1', 7, 12],
        'rip2': ['RIP2', 'RIP2.csv', 'RIP2', 7, 12],
        'pkci': ['PKCi', 'PKC-i.csv', 'PKC-i', 3, 8],
        'phgdh': ['PHGDH', 'PHGDH.csv', 'PHGDH', 7, 12],
        'rorg': ['RORg', 'RORg.csv', 'RORg', 7, 12],
        'ido1': ['IDO1', 'IDO1.csv', 'IDO1', 7, 12],
        'klk5': ['KLK5', 'KLK5.csv', 'KLK5', 7, 12],
        'notum':['Notum', 'Notum.csv', 'Notum', 7, 12],
        'eaat3': ['EAAT3', 'EAAT3.csv', 'EAAT3', 3, 8],
    }

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, feature_type=None):
        self.name = name.lower()
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        from rdkit import Chem

        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                x.append(x_map['degree'].index(atom.GetTotalDegree()))
                x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                x.append(x_map['num_radical_electrons'].index(
                    atom.GetNumRadicalElectrons()))
                x.append(x_map['hybridization'].index(
                    str(atom.GetHybridization())))
                x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                e.append(e_map['stereo'].index(str(bond.GetStereo())))
                e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name][0]}({len(self)})'
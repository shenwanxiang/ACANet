# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:40:35 2022

@author: wanxiang.shen@u.nus.edu
"""

import os
import os.path as osp
import pandas as pd
import numpy as np
import re
from rdkit import Chem
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)

from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_morgan_fingerprint(mol, radius=2, nBits=2048, device=None):
    """
    Calculates a Morgan fingerprint for a given RDKit Mol and returns a
    [1, nBits] PyTorch tensor of uint8 (0/1).
    """
    # 1) Compute the RDKit bit vector
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                        radius=radius,
                                                        nBits=nBits)
    # 2) Convert it to a flat NumPy array of shape (nBits,)
    arr = np.zeros((nBits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    # 3) Turn into a tensor of shape [1, nBits]
    tensor = torch.from_numpy(arr).unsqueeze(0)  # now shape [1, 2048]

    # 4) (Optional) move to device
    if device is not None:
        tensor = tensor.to(device)

    return tensor




class LSSNS(InMemoryDataset):
    r"""The benchmark datasets of low-sample size narrow scaffold inhibitors, 
    containing inhibitors from differnt drug targets of GPCRs, Kinases, Nuclear Receptors, Proteases, Transporters, and Other Enzymes.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (LSSNS().names.keys()).
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

    url = 'https://bidd-group.github.io/MPCD/dataset/LSSNS/{}'

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]

    names = {'ido1':['IDO1', 'IDO1.csv', 'IDO1', 7, 12],
             'plk1': ['PLK1', 'PLK1.csv', 'PLK1', 7, 12],
             'rip2': ['RIP2', 'RIP2.csv', 'RIP2', 7, 12],
             'braf': ['BRAF', 'BRAF.csv', 'BRAF', 7, 12],
             'usp7': ['USP7', 'USP7.csv', 'USP7', 7, 12],
             'phgdh': ['PHGDH', 'PHGDH.csv', 'PHGDH', 7, 12],
             'pkci': ['PKCi', 'PKCi.csv', 'PKCi', 3, 8],
             'rxfp1': ['RXFP1', 'RXFP1.csv', 'RXFP1', 7, 12],
             'mglur2': ['mGluR2', 'mGluR2.csv', 'mGluR2', 7, 12]
            }

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
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

        with open(self.raw_paths[0], 'r', encoding="utf-8") as f:
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
                
            fp_smiles = get_morgan_fingerprint(mol)
            fp_scaffold = get_morgan_fingerprint(MurckoScaffold.GetScaffoldForMol(mol))
            
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
                        smiles=smiles, fp_smiles=fp_smiles, fp_scaffold=fp_scaffold)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name][0]}({len(self)})'

    
    def smiles_to_data(self, smiles_list):

        data_list = []
        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            if mol is None:
                continue
            fp_smiles = get_morgan_fingerprint(mol)
            fp_scaffold = get_morgan_fingerprint(MurckoScaffold.GetScaffoldForMol(mol))
           
            
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

            x = torch.tensor(xs, dtype=torch.float).view(-1, 9)

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
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 3)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        smiles=smiles, fp_smiles = fp_smiles, fp_scaffold=fp_scaffold)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list


class HSSMS(LSSNS):
    r"""The benchmark datasets of higher-sample size diversity scaffold inhibitors, 
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
    url = 'https://bidd-group.github.io/MPCD/dataset/HSSMS/MoleculeACE_benchmark/{}'
    meta = 'https://bidd-group.github.io/MPCD/dataset/HSSMS/MoleculeACE_benchmark/metadata/datasets.csv'

    meta_table = pd.read_csv(meta)
    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {}
    for name in meta_table.Dataset.tolist():
        names.update({name.lower(): [name, name + '.csv', name, 0, 2]})

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, name, transform, pre_transform, pre_filter)

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

        with open(self.raw_paths[0], 'r', encoding="utf-8") as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            # smiles, raw_y, lgy, cliff, split
            smiles, y_nm, y_nm_lg, cliff, split = line

            cliff = int(cliff)
            y_nm = float(y_nm)
            y_nm_lg = float(y_nm_lg)
            y_pm_lg = -np.log10(y_nm*1e-9)  # convert to pIC50

            y_nm = torch.tensor([float(y_nm)], dtype=torch.float).view(1, -1)
            y_nm_lg = torch.tensor(
                [float(y_nm_lg)], dtype=torch.float).view(1, -1)
            y = torch.tensor([float(y_pm_lg)], dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            fp_smiles = get_morgan_fingerprint(mol)
            fp_scaffold = get_morgan_fingerprint(MurckoScaffold.GetScaffoldForMol(mol))
            
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

            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        cliff=cliff,
                        split=split,
                        y_nm=y_nm,
                        y_nm_lg=y_nm_lg,
                        smiles=smiles, 
                        fp_smiles=fp_smiles,
                       fp_scaffold=fp_scaffold
                       )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


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
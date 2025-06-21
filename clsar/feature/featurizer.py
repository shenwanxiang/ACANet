# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:14:54 2022

@author: Wanxiang.shen


"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
    
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

import numpy as np
import torch
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



    
class Gen39AtomFeatures(object):
    '''
    39 node features generation
    '''
    

    def __init__(self):

        self.in_channels = 39
        self.edge_dim = 10
        self.name = 'AT39'
    
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
        fp_smiles = get_morgan_fingerprint(mol)
        fp_scaffold = get_morgan_fingerprint(MurckoScaffold.GetScaffoldForMol(mol))

        xs = []
        for atom in mol.GetAtoms():
            symbol = [0.] * len(self.symbols)
            if atom.GetSymbol() in self.symbols:
                symbol[self.symbols.index(atom.GetSymbol())] = 1.
            else:
                symbol[self.symbols.index('other')] = 1.
                
            degree = [0.] * 6
            if atom.GetDegree() >= 6:
                degree[5] = 1.
            else:
                degree[atom.GetDegree()] = 1.

            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            
            hybridization = [0.] * len(self.hybridizations)
            hbdz = atom.GetHybridization()
            if hbdz in self.hybridizations:
                hybridization[self.hybridizations.index(hbdz)] = 1.
            else:
                hybridization[self.hybridizations.index('other')] = 1.

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
        data.fp_smiles = fp_smiles
        data.fp_scaffold = fp_scaffold
        
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

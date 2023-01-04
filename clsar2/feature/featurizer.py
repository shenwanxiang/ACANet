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

from .atomfeat import AtomFeaturizer
from. fingerprint import MorganFP, RdkitFP, EstateFP
from. fingerprint import  MACCSFP, PubChemFP, RGroupFP, FragmentFP


fp_dict = {'MorganFP':MorganFP, 'RdkitFP':RdkitFP, 
           'EstateFP':EstateFP, 'MACCSFP':MACCSFP, 
           'PubChemFP':PubChemFP, 'RGroupFP':RGroupFP, 'FragmentFP':FragmentFP}


class GenNodeEdgeFeatures115(object):
    
    '''
    115 various atom features:
    F = Featurizer(smiles)
    F.atom_types_feature:62
    F.atom_bonds_feature:5          
    F.atom_rings_feature:6          
    F.atom_lipinski_feature:3       
    F.atom_estate_indice:1          
    F.atom_descriptors_conribs:4    
    F.atom_env_feature:10            
    F.atom_inherent_feature:24       
    '''
    
    def __init__(self, fp_type = 'MorganFP', **kwargs):
        '''
        fp_type: 'MorganFP', 'RdkitFP', 'EstateFP', 'MACCSFP', 'PubChemFP', 'RGroupFP', 'FragmentFP'
        '''        
        self.in_channels = 115
        self.edge_dim = 10
        self.name = 'AT115'
        self.kwargs = kwargs
        self.fp_type = fp_type
        self.fp = fp_dict[fp_type](**kwargs)
        
        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
    
    
    def __call__(self, data):
        
        # Generate the 115 features.
        mol = Chem.MolFromSmiles(data.smiles)
        
        self.fp = self.fp(mol)
        data.fp = torch.tensor(self.fp.atom_fp_arr, dtype = torch.bool)
        
        AF = AtomFeaturizer(data.smiles)
        afs = {}
        afs.update(AF.atom_types_feature)
        afs.update(AF.atom_bonds_feature)
        afs.update(AF.atom_rings_feature)
        afs.update(AF.atom_lipinski_feature)
        afs.update(AF.atom_estate_indice)
        afs.update(AF.atom_descriptors_conribs)
        afs.update(AF.atom_env_feature)
        afs.update(AF.atom_inherent_feature)   

        df = pd.DataFrame(afs)
        df = df.fillna(0)
        x = df.clip(0,1).values #clip to 0-1 if not in this range
        data.x = torch.tensor(x)

        
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
    
    
class GenNodeEdgeFeatures39(object):
    '''
    AttentiveFP 39 node features generation
    '''

    def __init__(self, fp_type = 'MorganFP', **kwargs):
        '''
        fp_type: 'MorganFP', 'RdkitFP', 'EstateFP', 'MACCSFP', 'PubChemFP', 'RGroupFP', 'FragmentFP'
        '''
        self.in_channels = 39
        self.edge_dim = 10
        self.name = 'AT39'
        self.kwargs = kwargs
        self.fp_type = fp_type
        self.fp = fp_dict[fp_type](**kwargs)

        
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
        self.fp = self.fp(mol)
        data.fp = torch.tensor(self.fp.atom_fp_arr, dtype = torch.bool)
        
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
    
    
    

# class GenECFPFeatures(object):
    
#     '''
#     ECFP atom and bond feature
#     '''

#     def __init__(self, radius = 2, nBits = 1024):
        
#         self.radius = radius
#         self.nBits = nBits
#         self.name = 'ECFP'
#         self.in_channels = self.nBits
#         self.edge_dim = self.nBits
    
#     def __call__(self, data):
        
#         # Generate the 1024 bit features.
#         mol = Chem.MolFromSmiles(data.smiles)
#         fp, bitInfo = mol2fpbitInfo(mol, 
#                                     radius=self.radius, 
#                                     nBits=self.nBits)

#         OnbitIdx = list(bitInfo.keys())
#         atom_fp_arr = np.zeros((mol.GetNumAtoms(), self.nBits))
#         bond_fp_arr = np.zeros((mol.GetNumBonds(), self.nBits))


#         for idx in OnbitIdx:
#             bitpaths = bitInfo[idx]
#             atomsToUse, bondsToUse = bitpaths2AtomBondIdx(mol, bitpaths)
#             #highlight(mol, atomsToUse, bondsToUse)
#             atom_fp_arr[atomsToUse, idx] = 1.
#             bond_fp_arr[bondsToUse, idx] = 1.
    
#         data.x = torch.tensor(atom_fp_arr, dtype=torch.float) 
#         edge_attr = np.repeat(bond_fp_arr, repeats=2, axis=0)
#         data.edge_attr = torch.tensor(edge_attr,  dtype=torch.float) 
        
#         edge_indices = []
#         for bond in mol.GetBonds():
#             edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
#             edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
#         data.edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

#         return data

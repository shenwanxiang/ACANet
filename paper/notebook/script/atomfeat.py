#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 18:39:44 2019

@author: wanxiang.shen@u.nus.edu

@task: atomic feature calculator
"""


from meta import META_TABLE as MT
from meta import ORGANIC

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import EState
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
from rdkit.Chem.rdchem import HybridizationType

from collections import Counter
import pandas as pd
import numpy as np



#all_type_pattens = dict(Chem.EState.AtomTypes.esPatterns)
ATOMTYPES_ESTATE = ['ssBH', 'sssB', 'ssssB', 
                    'sCH3', 'dCH2', 'ssCH2', 'tCH', 'dsCH', 'aaCH', 'sssCH', 'ddC', 'tsC', 'dssC', 'aasC', 'aaaC', 'ssssC',
                    'sNH3', 'sNH2', 'ssNH2', 'dNH', 'ssNH', 'aaNH', 'tN', 'sssNH', 'dsN', 'aaN', 'sssN', 'ddsN', 'aasN', 'ssssN', 
                    'sOH', 'dO', 'ssO', 'aaO',
                    'sPH2', 'ssPH', 'sssP', 'dsssP', 'sssssP', 
                    'sSH', 'dS', 'ssS', 'aaS', 'dssS', 'ddssS']

ATOMTYPES_ORGANIC = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

#['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER']
 #list(HybridizationType.names.keys()) 
ATOMTYPES_HYBRID = ['S', 'SP', 'SP2', 'SP3']

#atom chirality type
ATOMTYPES_CHIRALITYT = ['R', 'S']   

#list(Chem.rdchem.BondType.names.keys())
BONDTYPES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

#N MemberedRings
RINGTYPES = [3, 4, 5, 6]

#atom inherent features
MT = MT[MT.columns[2:]]



class AtomBasicAttr:

    def __init__(self, atom):
        self._atom = atom
        self.idx = atom.GetIdx()
    
    @property
    def atomic_number(self):
        """ Atomic number of atom """
        return self._atom.GetAtomicNum()
    @property
    def degree(self):
        return self._atom.GetDegree()
    
    @property
    def explicit_valence(self):
        """ Explicit valence of atom """
        return self._atom.GetExplicitValence()
    @property
    def implicit_valence(self):
        """ Implicit valence of atom """
        return self._atom.GetImplicitValence()
    @property
    def valence(self):
        """ returns the valence of the atom """
        return self.explicit_valence + self.implicit_valence
    @property
    def num_implicit_hydrogens(self):
        """ Number of implicit hydrogens """
        return self._atom.GetNumImplicitHs()
    @property
    def num_explicit_hydrogens(self):
        """ Number of explicit hydrodgens """
        return self._atom.GetNumExplicitHs()
    @property
    def num_hydrogens(self):
        """ Number of hydrogens """
        return self.num_implicit_hydrogens + self.num_explicit_hydrogens
    @property
    def formal_charge(self):
        """ Formal charge of atom """
        return self._atom.GetFormalCharge()    
    @property
    def gasteiger_charge(self):
        """ Hacky way of getting gasteiger charge """
        res = self._atom.GetProp('_GasteigerCharge')
        return float(res)


def GetAtomInherentAttr(atomic_number):
    """atom lookup table feature
    parameters
    ----------------
    atomic_number:  int, such as 6 stands for Carbon atom, result of atom.GetAtomicNum() 
    """
    return MT.loc[atomic_number].to_dict()

        
        
def GetAtomBasicAttr(atom):    
    """atom basic feature"""
    f_dict = {}
    A = AtomBasicAttr(atom)
    f_dict['atomic_number'] = A.atomic_number
    f_dict['degree'] = A.degree
    f_dict['explicit_valence'] = A.explicit_valence
    f_dict['implicit_valence'] = A.implicit_valence
    f_dict['valence'] = A.valence
    f_dict['num_implicit_hydrogens'] = A.num_implicit_hydrogens
    f_dict['num_explicit_hydrogens'] = A.num_explicit_hydrogens
    f_dict['num_hydrogens'] = A.num_hydrogens
    f_dict['formal_charge'] = A.formal_charge
    f_dict['gasteiger_charge'] = A.gasteiger_charge
    return f_dict




class Featurizer(object):
    
    def __init__(self, smiles, include_hydrogen = False):
        """Atom featurizer
        parameters
        -----------------
        smiles: smiles string of a compund
        include_hydrogen: bool, if True, Hydrogen atom will be included. 
        
        attributes
        -----------------
        transform: transform to atom features with a dict format
        """
        
        mol = Chem.MolFromSmiles(smiles)
        if include_hydrogen:
            mol = AllChem.AddHs(mol)            
        
        #get coords
        AllChem.Compute2DCoords(mol)
        
        #get partical charges
        rdPartialCharges.ComputeGasteigerCharges(mol)
        
        #get conformer
        cmol = mol.GetConformer()
        
        #get estate atom types
        atom_types = EState.AtomTypes.TypeAtoms(mol)
        
        #get estate indices
        atom_estate_indices = EState.EState.EStateIndices(mol)
        
        #get atom Lipinski
        atom_HAcceptors = Lipinski._HAcceptors(mol)
        atom_HDonors = Lipinski._HDonors(mol)
        atom_Heteroatoms = Lipinski._Heteroatoms(mol)        
        atom_RotatableBonds = Lipinski._RotatableBonds(mol)
        
        
        #get atom logp, mr
        atom_logp_mr = Crippen._GetAtomContribs(mol)
        
        #get atom tpsa
        atom_tpsa = rdMolDescriptors._CalcTPSAContribs(mol)
        
        #get atom asa
        atom_asa = list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])
        
        self.smiles = smiles
        self.mol = mol
        self.cmol = cmol
        
        #self._atom_idx = list(range(mol.GetNumAtoms()))
        self._atom_types = atom_types
        
        self._atom_HAcceptors = atom_HAcceptors
        self._atom_HDonors = atom_HDonors
        self._atom_Heteroatoms = atom_Heteroatoms
        self._atom_RotatableBonds = atom_RotatableBonds
        
        self._atom_estate_indices = atom_estate_indices
        
        self._atom_logp_mr = atom_logp_mr
        self._atom_tpsa = atom_tpsa
        self._atom_asa = atom_asa
        
        

    def _atom_type_feature(self):
        """one-hot atom types feature: symbol types, estate types, hydrid types, chirality types """
        atom_type_f = {}
        
        # organic atom types
        atom_symbols = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        for atype in ATOMTYPES_ORGANIC:
            value = pd.Series(atom_symbols).isin([atype]).values
            atom_type_f['is_%s' % atype] = value*1.0 
        
        ## estate atom types
        atom_estate_types = []
        for x in self._atom_types:
            if len(x) == 0:
                atom_estate_types.append(None)
            else:
                atom_estate_types.append(x[0])
        for atype in ATOMTYPES_ESTATE:
            if len(atom_estate_types) == 0:
                value = False
            else:
                value = pd.Series(atom_estate_types).isin([atype]).values
            atom_type_f['is_%s' % atype] = value*1.0

            
        # hybridization type
        atom_htypes = [str(atom.GetHybridization()) for atom in self.mol.GetAtoms()]
        for atype in ATOMTYPES_HYBRID:
            n = str(atype)
            key = 'is_' + n + '_hybridized'
            value = pd.Series(atom_htypes).isin([n]).values*1.0
            atom_type_f[key] = value
            
            
        # chirality type 
        atom_chi_types = Chem.FindMolChiralCenters(self.mol)
        for ctype in ATOMTYPES_CHIRALITYT:
            value = np.zeros((self.mol.GetNumAtoms(),))
            key =  'is_chirality_%s' % ctype
            ctype_idx = []
            for actype in atom_chi_types:
                if ctype == actype[1]:
                    ctype_idx.append(actype[0])
            value[ctype_idx] = 1.0
            atom_type_f[key] = value            

        return atom_type_f
        

        
        
    def _lipinski_feature(self):
        """atom feature of is HBA, HBD, HETERO"""
        lps_feature = {}
        
        for key, sparse in zip(['is_h_acceptor', 'is_h_donor', 'is_hetero'],
                               [self._atom_HAcceptors, self._atom_HDonors, self._atom_Heteroatoms]):
            value = np.zeros((self.mol.GetNumAtoms(),))
            value[[x[0] for x in sparse]] = 1.0
            lps_feature[key] = value   

        return lps_feature
        
        
        
    def _bonds_feature(self):
        """atom bonds feature: number of RTB, SINGLE, DOUBLE, TRIPLE ..."""
        bonds_feature = {}
        x = []
        for i in self._atom_RotatableBonds:
            x.append(i[0])
            x.append(i[1])
        idx_rtb_num = dict((i, x.count(i)) for i in x)
        value = np.zeros((self.mol.GetNumAtoms(),))
        value[list(idx_rtb_num.keys())] = list(idx_rtb_num.values())
        bonds_feature['num_ROTATABLE_bonds'] = value           
        

        bond_types = []
        for atom in self.mol.GetAtoms():
            bonds = [str(bond.GetBondType()) for bond in atom.GetBonds()]
            bond_types.append(dict(Counter(bonds)))

        for bond in BONDTYPES:
            key = 'num_%s_bonds' % bond
            value = [at_bond.get(bond, 0) for at_bond in bond_types]
            bonds_feature[key] = np.array(value)
    
        return bonds_feature
    
    
        
    def _rings_feature(self):
        """atom rings feature: is in R,AR,3R,4R,5R 6R ..."""
        ring_features = {}
        
        value_is_r = []
        value_is_ar = []
        for atom in self.mol.GetAtoms():
            value_is_r.append(atom.IsInRing())
            value_is_ar.append(atom.GetIsAromatic())
            
        ring_features['is_in_Ring'] = np.array(value_is_r)*1.0        
        ring_features['is_in_AromaticRing'] = np.array(value_is_ar)*1.0     
        
        AtomRing = self.mol.GetRingInfo().AtomRings()
        for rt in RINGTYPES:
            key = 'is_in_%sMemberedRings' % rt
            value = np.zeros((self.mol.GetNumAtoms(),))
            rt_idx = []
            for ar in AtomRing:
                if len(ar) == rt:
                    rt_idx.extend(ar) 
            value[rt_idx] = 1.0
            ring_features[key] = value
            
        return ring_features
        

        
    def _descriptors_conribs(self):
        """contribs of logp, mr, tpsa, asa"""
        descrip_feature = {}
        logp = []
        mr = []
        for i in self._atom_logp_mr:
            logp.append(i[0])
            mr.append(i[1])
        descrip_feature['logp_contribs'] = np.array(logp)
        descrip_feature['mr_contribs'] = np.array(mr)
        descrip_feature['tpsa_contribs'] = np.array(self._atom_tpsa)        
        descrip_feature['asa_contribs'] = np.array(self._atom_asa)
        
        return descrip_feature
        

    def _position_feature(self):
        """atom coords x,y and vector length"""
        pos_f = {}
        pos_x = self.cmol.GetPositions()[:,0]
        pos_y = self.cmol.GetPositions()[:,1]
        pos_length =  np.array([self.cmol.GetAtomPosition(atom.GetIdx()).Length() for atom in self.mol.GetAtoms()])
        pos_f['pos_x'] = pos_x
        pos_f['pos_y'] = pos_y
        pos_f['pos_length'] = pos_length
        return pos_f
        
        
    def _basic_feature(self):
        """atom basic features"""
        features = []
        for atom in self.mol.GetAtoms():
            atom_f = GetAtomBasicAttr(atom)
            features.append(atom_f)
        
        df = pd.DataFrame(features)
        df = df[list(atom_f.keys())]
        res = {}
        for i in df.columns:
            res[i] = df[i].values
    
        return res
    
    
    def _inherent_feature(self):
        """atom inherent features: mass, density, melting_point..."""
        atomic_numbers = []
        for atom in self.mol.GetAtoms():
            atomic_numbers.append(atom.GetAtomicNum())
        
        uniq_num = list(np.unique(atomic_numbers))
        
        num_feat_map = {}
        for atomic_number in uniq_num:
            num_feat_map[atomic_number] = GetAtomInherentAttr(atomic_number)
            
        features = []
        for atomic_number in atomic_numbers:
            atom_f = num_feat_map.get(atomic_number)
            features.append(atom_f)
        df = pd.DataFrame(features)
        df = df[list(atom_f.keys())]

        res = {}
        for i in df.columns:
            res[i] = df[i].values
            
        return res
        

    @property
    def atomfeats(self):
        """Atom Features"""
        feature_dict = {}
        
        #atom types
        feature_dict.update(self._atom_type_feature())

        #atom bonds num feature
        feature_dict.update(self._bonds_feature())

        #atom ring feature
        feature_dict.update(self._rings_feature())

        #atom lipinski feature
        feature_dict.update(self._lipinski_feature())
        
        #atom estate indices
        feature_dict.update({'estate_indice': self._atom_estate_indices})
        
        #atom logp, mr, tpsa, asa contribs
        feature_dict.update(self._descriptors_conribs())
        
        #atom basic inherent feature
        feature_dict.update(self._basic_feature())        
                
        #atom inherent feature
        feature_dict.update(self._inherent_feature())
        return feature_dict

    


    
    
    
    
    
if __name__ == '__main__':
    
    F = Featurizer(smiles)
    smiles = 'CC(C)[C@@H](C(=O)N1CCC[C@H]1C(=O)Nc2ccc(cc2)[C@@H]3CC[C@H](N3c4ccc(cc4)C(C)(C)C)c5ccc(cc5)NC(=O)[C@@H]6CCCN6C(=O)[C@H](C(C)C)NC(=O)OC)NC(=O)OC'
    print(pd.DataFrame(F.atomfeats))
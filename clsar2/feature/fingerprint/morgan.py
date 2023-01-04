# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:14:54 2022

@author: Wanxiang.shen

Python code to highlight the ECFP fingerprints on a given molecule
"""

from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from copy import deepcopy
import numpy as np

from .bit import Bit, FP

  
def mol2fpbitInfo(mol, radius = 2, nBits = 1024):
    bitInfo = {}
    fp = GetMorganFingerprintAsBitVect(mol, 
                                       radius = radius, 
                                       nBits = nBits, 
                                       bitInfo = bitInfo)
    fp = np.array(fp.ToList())
    return fp, bitInfo


def bitinfo2AtomBondIdx(mol, bitInfo):
    # bitInfo: tuple, (20, 2), where 20 is the atom idx, 2 is the radius
    atomId,radius = bitInfo

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    bondenv = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomId)

    # get the atoms for highlighting
    atomsToUse = set((atomId, ))
    for b in bondenv:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())

    # enlarge the environment by one further bond
    bondToUse = set()
    for atom in atomsToUse:
        a = mol.GetAtomWithIdx(atom)
        for b in a.GetBonds():
            bidx = b.GetIdx()
            if bidx not in bondenv:
                bondToUse.add(bidx)
    bondToUse = list(bondToUse)
    bondToUse += bondenv
    return atomsToUse, bondToUse


def bitinfos2AtomBondIdx(mol, bitInfos):
    '''
    bitinfos: tuple, ((20,0), (10,0)), this is because the folding of the ECFP fingerprint.
    '''
    concat_atom2use = []
    concat_bond2use = []
    for bitInfo in bitInfos:
        atom2use, bond2use = bitinfo2AtomBondIdx(mol, bitInfo)
        concat_atom2use.extend(atom2use)
        concat_bond2use.extend(bond2use)
    atomsToUse = list(set(concat_atom2use))
    bondsToUse = list(set(concat_bond2use))
    return atomsToUse, bondsToUse



class MorganFP(FP):
    def __init__(self, nBits = 1024, radius = 2):
        self.radius = radius
        self.nBits = nBits
       
    
    def __call__(self, mol):

        fp_arr, bitinfodict = mol2fpbitInfo(mol, 
                                            radius=self.radius,
                                            nBits=self.nBits)
        bits = []
        atom_fp_arr = np.zeros((mol.GetNumAtoms(), self.nBits))
        bond_fp_arr = np.zeros((mol.GetNumBonds(), self.nBits))

        for idx, onbit in enumerate(fp_arr):
            name = 'ECFP%s' % idx
            if onbit:
                info = bitinfodict[idx]
                atomInUse, bondInUse = bitinfos2AtomBondIdx(mol, info)
                patt = Chem.PathToSubmol(mol, bondInUse)
                smarts = Chem.MolToSmarts(patt).replace('[#0]', '*')
                
                atom_fp_arr[atomInUse, idx] = 1.
                bond_fp_arr[bondInUse, idx] = 1.
            
            else:
                info = ()
                atomInUse = []
                bondInUse = []
                patt = None
                smarts = None
                
            descr = 'info is Atom idx and radius'
            onebitdict={'idx': idx, 'name': name, 'on': onbit, 'patt': patt, 
                        'smarts':smarts,'info': info, 'descr': descr, 
                        'atomInUse': atomInUse, 'bondInUse': bondInUse}
            
            B = Bit(onebitdict)
            bits.append(B)
            
        self.bits = bits
        self.fp_arr = fp_arr
        self.bitinfodict = bitinfodict
        self.atom_fp_arr = atom_fp_arr
        self.bond_fp_arr = bond_fp_arr
        
        return self

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:14:54 2022

@author: Wanxiang.shen

Python code to highlight the ECFP fingerprints on a given molecule
"""
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from copy import deepcopy
import numpy as np


def mol2fpbitInfo(mol, nBits = 1024, minPath = 1, maxPath = 5, **kwargs):
    bitInfo = {}
    fp = RDKFingerprint(mol, fpSize = nBits, 
                        minPath = minPath,
                        maxPath = maxPath,
                        bitInfo = bitInfo, **kwargs)
    fp = np.array(fp.ToList())
    return fp, bitInfo


def _bitinfo2AtomBondIdx(mol, bitinfo):
    # bitinfo: list, [1,2,3], bond index
    
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    
    # get the atoms for highlighting
    atomsToUse = set()
    for b in bitinfo:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())

    # enlarge the environment by one further bond
    bondToUse = bitinfo
    return atomsToUse, bondToUse


def bitinfos2AtomBondIdx(mol, bitinfos):
    '''
    bitinfos: tuple, ((20,0), (10,0)), this is because the folding of the ECFP fingerprint.
    '''
    concat_atom2use = []
    concat_bond2use = []
    for bitinfo in bitinfos:
        atom2use, bond2use = _bitinfo2AtomBondIdx(mol, bitinfo)
        concat_atom2use.extend(atom2use)
        concat_bond2use.extend(bond2use)
    atomsToUse = list(set(concat_atom2use))
    bondsToUse = list(set(concat_bond2use))
    return atomsToUse, bondsToUse



from .bit import Bit, FP

class RdkitFP(FP):
    def __init__(self, nBits = 1024, minPath = 1, maxPath = 5, **kwargs):
        self.nBits = nBits
        self.minPath = minPath
        self.maxPath = maxPath
        self.kwargs = kwargs
    
    
    def __call__(self, mol):

        fp_arr, bitinfodict = mol2fpbitInfo(mol, 
                                            nBits=self.nBits,
                                            minPath=self.minPath,
                                            maxPath = self.maxPath,
                                            **self.kwargs
                                           )
        bits = []
        atom_fp_arr = np.zeros((mol.GetNumAtoms(), self.nBits))
        bond_fp_arr = np.zeros((mol.GetNumBonds(), self.nBits))

        for idx, onbit in enumerate(fp_arr):
            name = 'RdkitFP%s' % idx
            if onbit:
                info = tuple(tuple(i) for i in bitinfodict[idx])
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
                
            descr = 'info is bond idx'
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

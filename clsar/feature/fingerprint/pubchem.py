#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:29:36 2019

@author: wanxiang.shen

@Note: The basic code is from PyBioMed & MolMap, with a major modification

https://www.ncbi.nlm.nih.gov/pubmed/29556758

these are SMARTS patterns corresponding to the PubChem fingerprints
https://web.cse.ohio-state.edu/~zhang.10631/bak/drugreposition/list_fingerprints.pdf
ftp://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem_fingerprints.txt

"""

import numpy as np
import pandas as pd
from rdkit import Chem
from itertools import chain
from rdkit import DataStructs
import os

_type = 'SMARTS-based'

file_path = os.path.dirname(__file__)
file_name = os.path.join(file_path, 'data', 'pubchembit.pkl')


def _InitBits(file_name):
    dfbit = pd.read_pickle(file_name)
    return dfbit

dfbit = _InitBits(file_name)
PubchemKeys = dfbit.patt.dropna().tolist() ## 733 bits, [0~114, 263-880]

'''
Please be noted that the bits of 115 to 262 have no substructure patterns.
'''


def b2a(mol, bond_idxs):
    atomsToUse = set()
    for b in bond_idxs:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())
    return tuple(atomsToUse)



def _map(temp, tempA, bits, bitinfodict, n = 0):

    ## three member rings
    R = 3
    IDXS = np.array([0, 7]) + n
    COUNT = [1, 2]
    for c, i in zip(COUNT, IDXS):
        if temp[R] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[R]
            else:
                bitinfodict[i] = tempA[R][:c]
                
    ## four member rings
    R = 4
    IDXS = np.array([14, 21]) + n
    COUNT = [1, 2]
    for c, i in zip(COUNT, IDXS):
        if temp[R] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[R]
            else:
                bitinfodict[i] = tempA[R][:c]

    ## five member rings
    R = 5
    IDXS = np.array([28, 35, 42, 49, 56]) + n
    COUNT = [1,2,3,4,5]
    for c, i in zip(COUNT, IDXS):
        if temp[R] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[R]
            else:
                bitinfodict[i] = tempA[R][:c]             

    ## six member rings
    R = 6
    IDXS = np.array([63, 70, 77, 84, 91]) + n
    COUNT = [1, 2, 3, 4, 5]
    for c, i in zip(COUNT, IDXS):
        if temp[R] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[R]
            else:
                bitinfodict[i] = tempA[R][:c]

    # seven member rings
    R = 7
    IDXS = np.array([98, 105])+n
    COUNT = [1, 2]
    for c, i in zip(COUNT, IDXS):
        if temp[R] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[R]
            else:
                bitinfodict[i] = tempA[R][:c]

    # eight member rings
    R = 8
    IDXS = np.array([112, 119]) + n
    COUNT = [1, 2]
    for c, i in zip(COUNT, IDXS):
        if temp[R] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[R]
            else:
                bitinfodict[i] = tempA[R][:c]
        
    # nine member rings
    R = 9
    IDXS = np.array([126]) + n
    COUNT = [1]
    for c, i in zip(COUNT, IDXS):
        if temp[R] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[R]
            else:
                bitinfodict[i] = tempA[R][:c]
                
    # ten member rings
    R = 10
    IDXS = np.array([133]) + n
    COUNT = [1]
    for c, i in zip(COUNT, IDXS):
        if temp[R] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[R]
            else:
                bitinfodict[i] = tempA[R][:c]

    return bits, bitinfodict
  
    
def func_1(mol, bits, bitinfodict):
    """ *Internal Use Only*
    Calculate PubChem Fingerprints （116-263)
    
    https://web.cse.ohio-state.edu/~zhang.10631/bak/drugreposition/list_fingerprints.pdf
    
    """
    
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    tempA = {3:(), 4:(), 5:(), 6:(), 7:(), 8:(), 9:(), 10:()}
    AllRingsAtom = mol.GetRingInfo().AtomRings()
    for ring in AllRingsAtom:
        ringSize.append(len(ring))
        for k,v in temp.items():
            if len(ring) == k:
                temp[k]+=1
                tempA[k] += (ring,)
           
    bits, bitinfodict = _map(temp, tempA, bits, bitinfodict, n = 0)
    return bits, bitinfodict


def func_2(mol, bits, bitinfodict):
    """ *Internal Use Only*
    saturated or aromatic carbon-only ring
    """
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    tempA = {3:(), 4:(), 5:(), 6:(), 7:(), 8:(), 9:(), 10:()}

    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
                    tempA[k] += (b2a(mol, ring),)
        ######## aromatic carbon-only     
        aromatic = True
        AllCarb = True
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() != 6 or EndAtom.GetAtomicNum() != 6:
                AllCarb = False
                break
        if aromatic == True and AllCarb == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
                    tempA[k] += (b2a(mol, ring),)
                    
    bits, bitinfodict = _map(temp, tempA, bits, bitinfodict, n = 1)
    return bits, bitinfodict


def func_3(mol, bits, bitinfodict):
    """ *Internal Use Only*
    saturated or aromatic nitrogen-containing
    """
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    tempA = {3:(), 4:(), 5:(), 6:(), 7:(), 8:(), 9:(), 10:()}

    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
                    tempA[k] += (b2a(mol, ring),)
        ######## aromatic nitrogen-containing    
        aromatic = True
        ContainNitro = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() == 7 or EndAtom.GetAtomicNum() == 7:
                ContainNitro = True
                break
        if aromatic == True and ContainNitro == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
                    tempA[k] += (b2a(mol, ring),)
                   
    bits, bitinfodict = _map(temp, tempA, bits, bitinfodict, n = 2)
    return bits, bitinfodict


def func_4(mol, bits, bitinfodict):
    """ *Internal Use Only*
    saturated or aromatic heteroatom-containing
    """
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize=[]
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    tempA = {3:(), 4:(), 5:(), 6:(), 7:(), 8:(), 9:(), 10:()}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
                    tempA[k] += (b2a(mol, ring),)
        ######## aromatic heteroatom-containing
        aromatic = True
        heteroatom = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if (aromatic == True) & (heteroatom == True):
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
                    tempA[k] += (b2a(mol, ring),)
                    
    bits, bitinfodict = _map(temp, tempA, bits, bitinfodict, n = 3)
    return bits, bitinfodict


def func_5(mol,bits, bitinfodict):
    """ *Internal Use Only*
    unsaturated non-aromatic carbon-only
    """
    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    tempA = {3:(), 4:(), 5:(), 6:(), 7:(), 8:(), 9:(), 10:()}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        Allcarb = True
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## allcarb
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() != 6 or EndAtom.GetAtomicNum() != 6:
                Allcarb = False
                break
        if (unsaturated == True) & (nonaromatic == True) & (Allcarb == True):
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
                    tempA[k] += (b2a(mol, ring),)
                    
    bits, bitinfodict = _map(temp, tempA, bits, bitinfodict, n = 4)
    return bits, bitinfodict


def func_6(mol,bits, bitinfodict):
    """ *Internal Use Only*
    unsaturated non-aromatic nitrogen-containing
    """
    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    tempA = {3:(), 4:(), 5:(), 6:(), 7:(), 8:(), 9:(), 10:()}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        ContainNitro = False
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## nitrogen-containing
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() == 7 or EndAtom.GetAtomicNum() == 7:
                ContainNitro = True
                break
        if unsaturated == True and nonaromatic == True and ContainNitro== True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
                    tempA[k] += (b2a(mol, ring),)
    bits, bitinfodict = _map(temp, tempA, bits, bitinfodict, n = 5)
    return bits, bitinfodict  



def func_7(mol, bits, bitinfodict):
    """ *Internal Use Only*
    unsaturated non-aromatic heteroatom-containing
    """
    ringSize=[]
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp={3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    tempA = {3:(), 4:(), 5:(), 6:(), 7:(), 8:(), 9:(), 10:()}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        heteroatom = False
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='SINGLE':
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name=='AROMATIC':
                nonaromatic = False
                break
        ######## heteroatom-containing
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if unsaturated == True and nonaromatic == True and heteroatom == True:
            ringSize.append(len(ring))
            for k,v in temp.items():
                if len(ring) == k:
                    temp[k]+=1
                    tempA[k] += (b2a(mol, ring),)
    bits, bitinfodict = _map(temp, tempA, bits, bitinfodict, n = 6)
    return bits, bitinfodict  



def func_8(mol, bits, bitinfodict):
    """ *Internal Use Only*
    aromatic rings or hetero-aromatic rings
    """
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp = {'aromatic':0,'heteroatom':0}
    tempA = {'aromatic':(), 'heteroatom':()}
    for ring in AllRingsBond:
        aromatic = True
        heteroatom = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name!='AROMATIC':
                aromatic = False
                break
        if aromatic==True:
            temp['aromatic']+=1
            tempA['aromatic'] += (b2a(mol, ring),)
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [1,6] or EndAtom.GetAtomicNum() not in [1,6]:
                heteroatom = True
                break
        if heteroatom==True:
            temp['heteroatom']+=1
            tempA['heteroatom'] += (b2a(mol, ring),)
            
    Ra = 'aromatic'
    IDXS = np.array([140, 142, 144, 146])
    COUNT = [1,2,3,4]
    for c, i in zip(COUNT, IDXS):
        if temp[Ra] >= c:
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[Ra]
            else:
                bitinfodict[i] = tempA[Ra][:c]  
                
                
    Rh = 'heteroatom'
    IDXS = np.array([141, 143, 145, 147])
    COUNT = [1,2,3,4]
    for c, i in zip(COUNT, IDXS):
        if (temp[Rh] >= c) & (temp[Ra] >= c):
            bits[i] = 1
            if c == COUNT[-1]:
                bitinfodict[i] = tempA[Rh]
            else:
                bitinfodict[i] = tempA[Rh][:c]  

    return bits, bitinfodict


def getOneBitinfo(mol, bitId):
    patt, count = PubchemKeys[bitId]
    info = mol.GetSubstructMatches(patt)
    # DONT NEED to be greater than the predefined count
    if len(info) <= count: 
        info = ()
    return info

def mol2fpbitInfoPart1(mol): #（0-114; 262-880)
    '''
    0-114; 262-880 bits
    Based on the Smart Patterns
    '''
    onbits = []
    infos = []
    nBits = len(PubchemKeys)
    for bitId in range(len(PubchemKeys)):
        info = getOneBitinfo(mol, bitId)
        if len(info) > 0:
            onbits.append(bitId)
            infos.append(info)
    num_fp = np.zeros((nBits, ))
    num_fp[onbits] = 1
    bitInfo = dict(zip(onbits, infos))
    return num_fp, bitInfo


def mol2fpbitInfoPart2(mol):# 115-262
    """
    115-262 bits, total 148 bits
    Calculate PubChem Fingerprints （116-263)
    """
    num_fp = np.array([0]*148)
    bitinfodict = {}
    for func in [func_1,func_2,func_3,func_4, 
                 func_5, func_6, func_7, func_8]:
        num_fp, bitinfodict=func(mol, num_fp, bitinfodict)
    return num_fp, bitinfodict


def mol2fpbitInfo(mol): # 0-880
    fp1, bitinfodict1 = mol2fpbitInfoPart1(mol)
    fp2, bitinfodict2 = mol2fpbitInfoPart2(mol)

    dfp1 = pd.Series(fp1).to_frame(name = 'onbit') #0-114; 262-880
    dfp2 = pd.Series(fp2).to_frame(name = 'onbit') #115-262

    dfp1 = dfp1.join(pd.Series(bitinfodict1).to_frame('bitinfo'))
    dfp2 = dfp2.join(pd.Series(bitinfodict2).to_frame('bitinfo'))

    idx = dfp1.index[:115].tolist()
    idx.extend(list(range(263, 881)))
    
    dfp1.index = idx
    dfp2.index = dfp2.index+115
    dfp = dfp1.append(dfp2).sort_index()

    fp = dfp.onbit.values
    bitinfodict = dfp.bitinfo.dropna().to_dict()
    return fp, bitinfodict


def bitinfos2AtomBondIdx(mol, bitinfos):
    '''
    bitinfos: tuple, ((1,2,3), (4,5,6)): tuple of tuple with atom idx
    '''
    concat_atom2use = list(chain(*bitinfos))
    concat_bond2use = []
    for atom in concat_atom2use:
        a = mol.GetAtomWithIdx(atom)
        for b in a.GetBonds():
            bidx = b.GetIdx()
            ba = b.GetBeginAtomIdx()
            ea = b.GetEndAtomIdx()            
            if (ba in concat_atom2use) & (ea in concat_atom2use):
                concat_bond2use.append(bidx) ###bug fix

    atomsToUse = list(set(concat_atom2use))
    bondsToUse = list(set(concat_bond2use))
    return atomsToUse, bondsToUse





#############################################

from bit import Bit, FP

class PubChemFP(FP):

    def __init__(self):
        self.nBits = len(dfbit)
        
    def __call__(self, mol):
        fp_arr, bitinfodict = mol2fpbitInfo(mol)
        bits = []
        atom_fp_arr = np.zeros((mol.GetNumAtoms(), self.nBits))
        bond_fp_arr = np.zeros((mol.GetNumBonds(), self.nBits))

        for idx, onbit in enumerate(fp_arr):
            ts = dfbit.iloc[idx]
            name = ts.name
            smarts = ts.smarts
            patt = ts.patt
            descr = ts.desc
            
            if onbit:
                info = bitinfodict[idx]
                atomInUse, bondInUse = bitinfos2AtomBondIdx(mol, info)
                atom_fp_arr[atomInUse, idx] = 1.
                bond_fp_arr[bondInUse, idx] = 1.
            else:
                info = ()
                atomInUse = []
                bondInUse = []

            onebitdict={'idx': idx, 'name': name, 'on': onbit, 'patt': patt, 
                        'smarts':smarts, 'info': info, 'descr': descr, 
                        'atomInUse': atomInUse, 'bondInUse': bondInUse}
            
            B = Bit(onebitdict)
            bits.append(B)
            
        self.bits = bits
        self.fp_arr = fp_arr
        self.bitinfodict = bitinfodict
        self.atom_fp_arr = atom_fp_arr
        self.bond_fp_arr = bond_fp_arr
        
        return self
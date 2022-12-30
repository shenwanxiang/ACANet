from rdkit.Chem.AllChem import GetMACCSKeysFingerprint
from rdkit.Chem.MACCSkeys import _pyGenMACCSKeys, _InitKeys, maccsKeys, smartsPatts
import numpy as np
from itertools import chain
import os
import pandas as pd


def _InitBits(file_name):
    dfbit = pd.read_pickle(file_name)
    return dfbit

maccsKeys = [(None, 0)] * len(smartsPatts.keys())
_InitKeys(maccsKeys, smartsPatts)

file_name = os.path.join(os.path.dirname(__file__), 'data', 'maccsbit.pkl')
dfbit = _InitBits(file_name)



def getOneBitinfo(mol, bitId):
    #bitId: 0~165, total 166 bits
    special_cases = (0, 124, 165)
    #print(bitId)
    if bitId not in special_cases:
        patt, count = maccsKeys[bitId]
        info = mol.GetSubstructMatches(patt)
        # DONT NEED to be greater than the predefined count
        # if len(info) <= count: 
        #     info = ()
    else:
        if bitId == 124:
            # special case: num aromatic rings > 1
            info = (tuple(at.GetIdx() for at in mol.GetAromaticAtoms()),)
        elif bitId == 165:
            fragAssignment = []
            mol_frags = Chem.GetMolFrags(mol, asMols = True, frags=fragAssignment)
            info = (tuple(np.argwhere(fragAssignment).reshape(-1,)), )
        else:
            info = ()
    return info

def mol2fpbitInfo(mol):
    bitInfo = {}
    fp = GetMACCSKeysFingerprint(mol)
    onbits = np.array(fp.GetOnBits())-1
    
    num_fp = np.array(fp)[1:]
    infos = [getOneBitinfo(mol, bitId) for bitId in onbits]
    bitInfo = dict(zip(onbits, infos))
    return num_fp, bitInfo


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

class MACCSFP(FP):

    def __init__(self):
        self.nBits = len(maccsKeys)
        
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
            descr = 'info is atom idx'
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

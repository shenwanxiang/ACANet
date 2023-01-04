import numpy
from rdkit.Chem import Fragments
import numpy as np
from rdkit import Chem, RDConfig
from itertools import chain
import os

from .bit import Bit, FP

'''
SMARTS-patt based
'''

defaultPatternFileName = os.path.join(
    RDConfig.RDDataDir, 'FragmentDescriptors.csv')


def _LoadRdkitFragPatts(fileName):
    patts = []
    with open(fileName, 'r') as inF:
        for line in inF.readlines():
            if len(line) and line[0] != '#':
                splitL = line.split('\t')
                if len(splitL) >= 3:
                    name = splitL[0]
                    descr = splitL[1]
                    sma = splitL[2]
                    descr = descr.replace('"', '')
                    patt = Chem.MolFromSmarts(sma)
                    res = (patt, name, sma, descr)
                    patts.append(res)
    return patts

## init the patterns
frgPatterns = _LoadRdkitFragPatts(defaultPatternFileName)
fragKeys = dict(zip(range(len(frgPatterns)), frgPatterns))



def getOneBitinfo(mol, bitId):
    #bitId: 0~78, total 79 bits
    patt, name, sma, descr = fragKeys[bitId]
    #print(bitId)
    info = mol.GetSubstructMatches(patt)
    return info

              
    
def mol2fpbitInfo(mol):
    bitInfo = {}
    num_fp = []
    for idx, patt_ in fragKeys.items():
        patt, name, sma, descr = patt_
        if mol.HasSubstructMatch(patt):
            num_fp.append(1)
        else:
            num_fp.append(0)
    
    num_fp = np.array(num_fp)
    onbits = np.argwhere(num_fp).reshape(-1,)
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




class FragmentFP(FP):

    def __init__(self):
        self.nBits = len(fragKeys)
        
        
    def __call__(self, mol):
        fp_arr, bitinfodict = mol2fpbitInfo(mol)
        bits = []
        atom_fp_arr = np.zeros((mol.GetNumAtoms(), self.nBits))
        bond_fp_arr = np.zeros((mol.GetNumBonds(), self.nBits))

        for idx, onbit in enumerate(fp_arr):
            name = 'Fragment%s' % idx
            patt, smi, smarts, descr = fragKeys[idx] 
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

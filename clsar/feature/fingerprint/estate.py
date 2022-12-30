import numpy
from rdkit.Chem.EState import EStateIndices, Fingerprinter
from rdkit.Chem.EState import AtomTypes, TypeAtoms
import numpy as np
from rdkit import Chem
from itertools import chain

'''
SMARTS-patt based
'''

## init the patterns
AtomTypes.BuildPatts()
esPatterns = AtomTypes.esPatterns #('dCH2', <rdkit.Chem.rdchem.Mol at 0x7fea8686f200>),
esKeys = dict(zip(range(len(esPatterns)), esPatterns))

def getOneBitinfo(mol, bitId):
    #bitId: 0~78, total 79 bits
    name, patt = esKeys[bitId]
    #print(bitId)
    info = mol.GetSubstructMatches(patt)
    return info

              
    
def mol2fpbitInfo(mol):
    bitInfo = {}
    num_fp = Fingerprinter.FingerprintMol(mol)[0].astype(bool)*1
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




from bit import Bit, FP

class EstateFP(FP):

    def __init__(self):
        self.nBits = len(esKeys)
        
    def __call__(self, mol):
        fp_arr, bitinfodict = mol2fpbitInfo(mol)
        bits = []
        atom_fp_arr = np.zeros((mol.GetNumAtoms(), self.nBits))
        bond_fp_arr = np.zeros((mol.GetNumBonds(), self.nBits))

        for idx, onbit in enumerate(fp_arr):
            name = 'Estate%s' % idx
            smarts, patt = esKeys[idx]
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

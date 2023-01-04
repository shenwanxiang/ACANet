import pandas as pd
from rdkit import Chem
import numpy as np
from itertools import chain
import os



file_path = os.path.dirname(__file__)


def _InitKeys():
    # df = pd.read_csv('/home/shenwanxiang/Research/xAI-drug-discovery-flow/R-group-replacement/R-group Database/edges_final_network.csv')
    # g1 =  df.groupby('source').size().sort_values(ascending=False)
    # g2 =  df.groupby('target').size().sort_values(ascending=False)
    # d1 = g1.to_dict()
    # d2 = g2.to_dict()
    # d2.update(d1)
    # info = pd.Series(d2).sort_values(ascending=False)
    info = pd.read_json(os.path.join(file_path, 'data', 'R_group.json'), typ = 'Series')
    patt_list = info.index.map(lambda x:Chem.MolFromSmarts(x)).to_list()
    smi_list = info.index.to_list()
    freq_list = info.to_list()
    RGroupKeys = []
    for patt, smi, freq in zip(patt_list, smi_list, freq_list):
        RGroupKeys.append([patt, smi, freq])
    return RGroupKeys


def _InitKeys2():
    # df = pd.read_csv('/home/shenwanxiang/Research/xAI-drug-discovery-flow/R-group-replacement/R-group Database/edges_final_network.csv')
    # g1 =  df.groupby('source').size().sort_values(ascending=False)
    # g2 =  df.groupby('target').size().sort_values(ascending=False)
    # d1 = g1.to_dict()
    # d2 = g2.to_dict()
    # d2.update(d1)
    # info = pd.Series(d2).sort_values(ascending=False)
    info = pd.read_json(os.path.join(file_path,'data', 'R_group.json'), typ = 'Series')
    patt_list = info.index.map(lambda x:Chem.MolFromSmarts(x.replace('*', ''))).to_list()
    smi_list = info.index.map(lambda x:x.replace('*', '')).to_list()
    freq_list = info.to_list()
    RGroupKeys = []
    for patt, smi, freq in zip(patt_list, smi_list, freq_list):
        RGroupKeys.append([patt, smi, freq])
    return RGroupKeys

RGroupKeys =  _InitKeys2()
dfbit = pd.DataFrame(RGroupKeys, columns = ['patt', 'smarts', 'desc'])
dfbit['idx'] = dfbit.index
dfbit['name'] = 'RGroup'+dfbit.index.astype(str)
dfbit = dfbit[['idx', 'name', 'smarts', 'desc', 'patt']]



def getOneBitinfo(mol, bitId):
    patt, smi, freq = RGroupKeys[bitId]
    info = mol.GetSubstructMatches(patt)
    return info


def mol2fpbitInfo(mol, nBits = 2048):
    RGroupKeysToUse = RGroupKeys[:nBits]
    onbits = []
    infos = []
    for bitId in range(len(RGroupKeysToUse)):
        info = getOneBitinfo(mol, bitId)
        if len(info) > 0:
            onbits.append(bitId)
            infos.append(info)
    num_fp = np.zeros((nBits, ))
    num_fp[onbits] = 1
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

from .bit import Bit, FP

class RGroupFP(FP):

    def __init__(self, nBits = 2048):
        self.nBits = nBits
        self._dfbit = dfbit.head(nBits)
        
        
    def __call__(self, mol):
        fp_arr, bitinfodict = mol2fpbitInfo(mol, nBits = self.nBits)
        bits = []
        atom_fp_arr = np.zeros((mol.GetNumAtoms(), self.nBits))
        bond_fp_arr = np.zeros((mol.GetNumBonds(), self.nBits))

        for idx, onbit in enumerate(fp_arr):
            ts = self._dfbit.iloc[idx]
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
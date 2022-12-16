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


def mol2fpbitInfo(mol, radius = 2, nBits = 1024):
    bitInfo = {}
    fp = GetMorganFingerprintAsBitVect(mol, 
                                       radius = radius, 
                                       nBits = nBits, 
                                       bitInfo = bitInfo)
    return fp, bitInfo


def bitpath2AtomBondIdx(mol, bitPath):
    # bitPath: tuple, (20, 2), where 20 is the atom idx, 2 is the radius
    atomId,radius = bitPath

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


def bitpaths2AtomBondIdx(mol, bitpaths):
    '''
    bitpaths: tuple, ((20,0), (10,0)), this is because the folding of the ECFP fingerprint.
    '''
    concat_atom2use = []
    concat_bond2use = []
    for bitpath in bitpaths:
        atom2use, bond2use = bitpath2AtomBondIdx(mol, bitpath)
        concat_atom2use.extend(atom2use)
        concat_bond2use.extend(bond2use)
    atomsToUse = list(set(concat_atom2use))
    bondsToUse = list(set(concat_bond2use))
    return atomsToUse, bondsToUse



def showAtomIdx(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomLabel',str(atom.GetIdx()))
    return Draw.MolToImage(mol, includeAtomNumbers=True)


def highlight(mol, 
              atomsToUse = None, 
              bondToUse = None, 
              width = 300, 
              height =300,
              highlightAtomColors = None,
              highlightBondColors = None,
              highlightAtomRadii = None,**kwargs):
    d = rdMolDraw2D.MolDraw2DSVG(width, height) # or MolDraw2DCairo to get PNGs
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atomsToUse,
                                       highlightBonds=bondToUse, 
                                       highlightBondColors = highlightBondColors,
                                       highlightAtomColors = highlightAtomColors,
                                       highlightAtomRadii = highlightAtomRadii,
                                       **kwargs)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return SVG(svg)



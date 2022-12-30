# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:14:54 2022

@author: Wanxiang.shen

Python code to highlight the ECFP fingerprints on a given molecule
"""
from rdkit.Chem.rdmolops import RDKFingerprint
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from copy import deepcopy


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
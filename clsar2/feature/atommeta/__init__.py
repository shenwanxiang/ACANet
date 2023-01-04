#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Tue Oct 8 10:21:14 2019

@author: wanxiang.shen@u.nus.edu

this meta data is original from:

* 1) mendeleev: https://github.com/lmmentel/mendeleev/blob/master/mendeleev/tables.py
* 2) scikit-chem: https://github.com/lewisacidic/scikit-chem/tree/master/skchem/resource
"""


import os
import pandas as pd

def GetMeta(*args):

    """ passes a file path for a data resource specified """

    return os.path.join(os.path.dirname(__file__), *args)

META_TABLE = pd.read_csv(GetMeta('atom_attr_data.csv'), index_col=0)
ORGANIC = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']


amt = META_TABLE
mt_org = amt[amt.symbol.isin(ORGANIC)]
mt_data = amt[amt.columns[2:]]
smax = mt_org[mt_org.columns[2:]].max()
smin = mt_org[mt_org.columns[2:]].min()
Scaled_ATOMIC_META_TABLE = (mt_data- smin) / (smax - smin) #scaled by ORGANIC atoms


__all__ = [
    'GetMeta', 'META_TABLE', 'ORGANIC', 'Scaled_ATOMIC_META_TABLE'
]

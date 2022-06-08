# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:47:00 2022

@author: wanxiang.shen@u.nus.edu
"""


import os
import pandas as pd
import numpy as np
import torch
from clsar.model import ACNet_GCN, ACNet_GIN, ACNet_GAT, ACNet_PNA
from clsar.dataset import LSSNS, HSSMS

# computation device
device = 'cpu'#('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")


def GetModelFile(*args):
    """ passes a file path for a data resource specified """
    return os.path.join(os.path.dirname(__file__), *args)


## make prediction
@torch.no_grad()
def _predict(smiles_list, transformer, model):
    loader = transformer(smiles_list)
    res = []
    for data in loader:
        data = data.to(device)
        E = model.eval()
        out = E(data.x.float(), data.edge_index, data.edge_attr, data.batch)
        res.append(out.cpu().numpy())
    return res


def predict(smiles_list):
    outputs = []
    for i in range(5):
        mfile = GetModelFile('trained_model/model_%s.pth' % (i+1))
        checkpoint = torch.load(mfile, map_location=torch.device('cpu'))
        #print(checkpoint['epoch'])
        model = ACNet_GAT(**checkpoint['model_args']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        transformer = checkpoint['data_transformer']
        res = _predict(smiles_list, transformer, model)
        outputs.append(res)

    all_prediction_mean = []
    for i in range(len(smiles_list)):
        one_prediction = [outputs[j][i] for j in range(5)]
        one_prediction_mean = np.mean(one_prediction, axis=0)
        all_prediction_mean.append(one_prediction_mean)
    return all_prediction_mean
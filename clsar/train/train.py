from math import sqrt
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy.stats import pearsonr
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn.models import AttentiveFP



from clsar.model.loss import ACALoss, get_best_cliff
from clsar.model import ACANet_PNA, get_deg  # model
from clsar.feature import GenAttentiveFeatures  # feature
from clsar.dataset import LSSNS  # dataset
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', font_scale=2)



torch.cuda.set_device(gpuid)


def _train(train_loader, device, optimizer, model, aca_loss):
    total_examples = 0
    total_loss = 0
    total_tsm_loss = 0
    total_reg_loss = 0
    n_triplets = []
    n_pos_triplets = []
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        predictions, embeddings = model(data.x.float(), data.edge_index,
                                        data.edge_attr, data.batch)

        loss_out = aca_loss(labels=data.y,
                            predictions=predictions,
                            embeddings=embeddings)

        loss, reg_loss, tsm_loss, n, n_pos = loss_out

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_tsm_loss += float(tsm_loss) * data.num_graphs
        total_reg_loss += float(reg_loss) * data.num_graphs
        total_examples += data.num_graphs

        n_triplets.append(int(n))
        n_pos_triplets.append(int(n_pos))

    train_loss = total_loss / total_examples
    total_tsm_loss = total_tsm_loss / total_examples
    total_reg_loss = total_reg_loss / total_examples
    n_triplets = int(sum(n_triplets) / (i+1))
    n_pos_triplets = int(sum(n_pos_triplets) / (i+1))

    return train_loss, total_tsm_loss, total_reg_loss, n_triplets, n_pos_triplets


pre_transform = GenAttentiveFeatures()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# model HPs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deg = get_deg(train_dataset)
pub_args = {'in_channels':pre_transform.in_channels, 
            'edge_dim':pre_transform.edge_dim, 
            'hidden_channels':64, 
            'out_channels':1, 
            'num_layers':10, 
            'dropout_p':0.1, 
            'batch_norms':None}

model = ACANet_PNA(**pub_args,
                   aggregators = ['mean', 'min', 'max', 'sum'],
                   scalers = ['identity','amplification', 'attenuation'],
                   deg=deg).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=10**-5)
# initialize SaveBestModel class

model_save_name = 'model_%s.pth' % dataset_name
saver = SaveBestModel(data_transformer = dataset.smiles_to_data, 
                      save_dir = result_save_dir, save_name = model_save_name)
aca_loss = ACALoss(alpha=10, cliff_lower=1, cliff_upper=1)

history = []
for epoch in tqdm(range(epochs), desc = 'epoch', ascii=True):
    train_loss, train_tsm, train_reg, train_n_triplets, train_n_pos_triplets = _train(train_loader, device, optimizer, model, aca_loss)
    saver(train_loss, epoch, model, optimizer)
    history.append([train_loss, train_tsm, train_reg, train_n_triplets, train_n_pos_triplets])
    
saver.save()
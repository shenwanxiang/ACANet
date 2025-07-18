from math import sqrt
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from rdkit import Chem
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import global_mean_pool, global_max_pool
%matplotlib inline
#A100 80GB
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances

gpuid = 1
torch.cuda.set_device(gpuid)
print(torch.cuda.current_device())

from clsar.dataset import LSSNS, HSSMS
from clsar.feature import Gen39AtomFeatures
from clsar.model.model import ACANet_PNA, get_deg, _fix_reproducibility # model
from clsar.model.loss import ACALoss, get_best_cliff
from mysaver import SaveBestModel
_fix_reproducibility(42)


def label_incoherence_index(X, y, k=5):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nbrs.kneighbors(X)
    
    incoherence_values = []
    for i in range(X.shape[0]):
        neighbors_labels = y[indices[i, 1:]]
        local_incoherence = np.mean(np.abs(neighbors_labels - y[i]))
        incoherence_values.append(local_incoherence)
    return np.mean(incoherence_values)



def get_structure_label_coherence(smiles_list, ys, k = 5):
    
    dim = 2048
    from rdkit.Chem import AllChem
    from tqdm import tqdm
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    ECFP4_fps = [AllChem.GetMorganFingerprintAsBitVect(x,2,dim) for x in tqdm(mols, ascii=True)]
    ecfps = np.array([list(fp) for fp in ECFP4_fps])
    lii = label_incoherence_index(ecfps, ys, k = k)
    #print("基于结构的标签不连贯性:", lii)
    return lii
    
def train(train_loader, model, optimizer, aca_loss):

    total_examples = 0
    total_loss =  0    
    total_tsm_loss = 0
    total_reg_loss = 0   
    n_triplets = []
    n_pos_triplets = []
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        predictions, embeddings = model(data.x.float(), data.edge_index, 
                                        data.edge_attr, data.batch)
        
        loss_out = aca_loss(labels = data.y, 
                            predictions = predictions,
                            embeddings = embeddings)
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

@torch.no_grad()
def test(test_loader, model, aca_loss):
    model.eval()
    total_examples = 0
    total_loss = 0
    total_tsm_loss = 0
    total_reg_loss = 0
    n_triplets = []
    n_pos_triplets = []
    mse = []
    for i, data in enumerate(test_loader):
        data = data.to(device)
        predictions, embeddings = model(data.x.float(), data.edge_index,
                                        data.edge_attr, data.batch)
        loss_out = aca_loss(labels=data.y,
                            predictions=predictions,
                            embeddings=embeddings)

        loss, reg_loss, tsm_loss, n, n_pos = loss_out

        total_loss += float(loss) * data.num_graphs
        total_tsm_loss += float(tsm_loss) * data.num_graphs
        total_reg_loss += float(reg_loss) * data.num_graphs
        total_examples += data.num_graphs

        n_triplets.append(int(n))
        n_pos_triplets.append(int(n_pos))

        mse.append(F.mse_loss(predictions, data.y, reduction='none').cpu())

    test_loss = total_loss / total_examples
    total_tsm_loss = total_tsm_loss / total_examples
    total_reg_loss = total_reg_loss / total_examples
    n_triplets = int(sum(n_triplets) / (i+1))
    n_pos_triplets = int(sum(n_pos_triplets) / (i+1))
    
    test_rmse = float(torch.cat(mse, dim=0).mean().sqrt())
    
    return test_loss, total_tsm_loss, total_reg_loss, n_triplets, n_pos_triplets, test_rmse

@torch.no_grad()
def predict(data_loader, model):
    #test_loader, device, model):
    embeds = []
    preds = []
    ys = []
    smiles_list = []
    model.eval()
    for data in data_loader:
        
        data = data.to(device)
        predictions, embeddings = model(data.x.float(), data.edge_index,
                                        data.edge_attr, data.batch)
        embeds.append(embeddings)
        preds.append(predictions)
        ys.append(data.y)
        smiles_list.extend(data.smiles)
        
    embeddings = torch.concat(embeds, axis=0).cpu().numpy()
    predictions = torch.concat(preds, axis=0).cpu().numpy()   
    ys = torch.concat(ys, axis=0).cpu().numpy() 
    return embeddings, predictions, ys, smiles_list
    
def Test_performance(alpha=1.0, cl=1, cu=1):
    model = ACANet_PNA(**pub_args, deg=deg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    aca_loss = ACALoss(alpha=alpha, cliff_lower=cl, cliff_upper=cu)

    model_save_name = f'model_{dataset_name}_{alpha}.pth'
    saver = SaveBestModel(
        data_transformer=dataset.smiles_to_data,
        save_dir=result_save_dir,
        save_name=model_save_name
    )

    history = []
    for epoch in range(0, epochs + 1):
        if epoch == 0:
            # 初始化评估
            _, _, _, _, train_n_pos_triplets, train_rmse = test(train_loader, model, aca_loss)
            _, _, _, _, val_n_pos_triplets,   val_rmse   = test(val_loader,   model, aca_loss)
            _, _, _, _, test_n_pos_triplets,  test_rmse  = test(test_loader,  model, aca_loss)

            # **在这里把 epoch=0 的结果也交给 saver 记录**
            saver(val_rmse, epoch, model, optimizer)

            print(f'Epoch: 000 (init), Train RMSE: {train_rmse:.4f}, '
                  f'Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}')

            history.append({
                'Epoch': 0,
                'train_loss': None,
                'train_triplet_loss': None,
                'train_mae_loss': None,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'train_rmse': train_rmse,
                'n_triplets': None,
                'n_pos_triplets': None,
                'train_n_pos_triplets': train_n_pos_triplets,
                'val_n_pos_triplets': val_n_pos_triplets,
                'test_n_pos_triplets': test_n_pos_triplets
            })
        else:
            # 正常训练 + 测试
            train_loss, tsm_loss, reg_loss, n_triplets, n_pos_triplets = train(
                train_loader, model, optimizer, aca_loss
            )
            _, _, _, _, train_n_pos_triplets, train_rmse = test(train_loader, model, aca_loss)
            _, _, _, _, val_n_pos_triplets,   val_rmse   = test(val_loader,   model, aca_loss)
            _, _, _, _, test_n_pos_triplets,  test_rmse  = test(test_loader,  model, aca_loss)

            print(
                f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, '
                f'tsm_loss: {tsm_loss:.4f}, reg_loss: {reg_loss:.4f}, '
                f'n_pos_triplets: {n_pos_triplets:03d}; '
                f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}'
            )

            saver(val_rmse, epoch, model, optimizer)

            history.append({
                'Epoch': epoch,
                'train_loss': train_loss,
                'train_triplet_loss': tsm_loss,
                'train_mae_loss': reg_loss,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'train_rmse': train_rmse,
                'n_triplets': n_triplets,
                'n_pos_triplets': n_pos_triplets,
                'train_n_pos_triplets': train_n_pos_triplets,
                'val_n_pos_triplets': val_n_pos_triplets,
                'test_n_pos_triplets': test_n_pos_triplets
            })

    # 训练／评估结束后保存最优模型
    saver.save()
    return pd.DataFrame(history)



def split_dataset_by_flag(dataset, split_attr='split'):
    from torch.utils.data import Subset
    split = getattr(dataset.data, split_attr)
    train_indices = [i for i, s in enumerate(split) if s == 'train']
    test_indices = [i for i, s in enumerate(split) if s == 'test']
    return Subset(dataset, train_indices), Subset(dataset, test_indices)




############################################################################
Dataset =  HSSMS #LSSNS
dataset_list = pd.DataFrame.from_dict(Dataset.names).loc[0].tolist()

for epochs in [0, 1, 10, 100, 200, 400, 600, 800, 1000]:
    for dataset_name in dataset_list:
        batch_size = 128
        lr = 1e-4
        result_save_dir = './emb_results_%s/%s' % (epochs, dataset_name)
        
        pre_transform = Gen39AtomFeatures()
        in_channels = pre_transform.in_channels
        path = result_save_dir
        
        ## model HPs
        pub_args = {'in_channels':pre_transform.in_channels, 
                    'edge_dim':pre_transform.edge_dim,
                    'convs_layers': [64, 128, 256, 512],   
                    'dense_layers': [256, 128, 64], 
                    'out_channels':1, 
                    'aggregators': ['mean', 'min', 'max', 'sum','std'],
                    'scalers':['identity', 'amplification', 'attenuation'] ,
                    'dropout_p': 0}
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        dataset = Dataset(path, name=dataset_name, pre_transform=pre_transform)
        train_dataset, test_dataset = split_dataset_by_flag(dataset)
        val_dataset = test_dataset
        # train, valid, test splitting
    
        if epochs == 0:
            c = 1
        else:
            from clsar.model.loss import get_best_cliff_batch
            c = get_best_cliff_batch(train_dataset, device=device, iterations=1)
        
        ############################################################################
        res1 = []
        res2 = []
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = val_loader
        deg = get_deg(train_dataset)
        # With AC-Awareness ($\alpha = 1$)
        df1 = Test_performance(alpha=1.0, cl = c, cu = c)
        # Without AC-Awareness ($\alpha = 0$)
        df2 = Test_performance(alpha=0.0, cl = c, cu = c)
        res1.append(df1)
        res2.append(df2)
        df1 = pd.concat(res1)
        df2 = pd.concat(res2)
        df1.to_csv(os.path.join(result_save_dir,'with_aca.csv'))
        df2.to_csv(os.path.join(result_save_dir,'without_aca.csv'))
        
        ############################################################################
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ['#FFE699','#00B0F0']
        n1 = r'With AC-awareness ($\mathcal{L}_{mae} + \mathcal{L}_{tsm}$)'
        n2 = r'Without AC-awareness ($\mathcal{L}_{mae}$)'
        y = 'test_rmse'
        dfp = df2.groupby('Epoch').mean()[y].to_frame(name = n2).join(df1.groupby('Epoch').mean()[y].to_frame(name = n1)).rolling(1).mean()
        dfp_std = df2.groupby('Epoch').std()[y].to_frame(name = n2).join(df1.groupby('Epoch').std()[y].to_frame(name = n1)).rolling(1).mean()
        dfp.plot(lw = 2, ax=ax,color = colors, alpha =1)
        ax.fill_between(dfp.index, (dfp - dfp_std)[n1], (dfp + dfp_std)[n1], color=colors[1], alpha=0.2)
        ax.fill_between(dfp.index, (dfp - dfp_std)[n2], (dfp + dfp_std)[n2], color=colors[0], alpha=0.2)
        ax.set_ylim(0.40, 1.5)
        ax.set_ylabel('Test RMSE')
        ax.set_xlabel('epochs')
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim(1,epochs)
        ax.tick_params(left='off', labelleft='on', labelbottom='on', bottom = 'off',  pad=.5,)
        fig.savefig(os.path.join(result_save_dir,'Test_RMSE.svg') , bbox_inches='tight', dpi=400) 
        fig.savefig(os.path.join(result_save_dir,'Test_RMSE.pdf'), bbox_inches='tight', dpi=400) 
        
        
        
        ############################################################################
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        deg = get_deg(train_dataset)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_no_aca = ACANet_PNA(**pub_args, deg=deg).to(device)  
        model_wi_aca = ACANet_PNA(**pub_args, deg=deg).to(device)  
        
        mfile_no_aca = os.path.join(result_save_dir, 'model_%s_%s.pth' % (dataset_name, 0.0))
        model_no_aca.load_state_dict(torch.load(mfile_no_aca, map_location=device)['model_state_dict'])
        
        mfile_wi_aca = os.path.join(result_save_dir, 'model_%s_%s.pth' % (dataset_name, 1.0))
        model_wi_aca.load_state_dict(torch.load(mfile_wi_aca, map_location=device)['model_state_dict'])
        
        
        p_res = []
        for model, aca_used in zip([model_no_aca, model_wi_aca], ['wo_aca', 'wi_aca']):
            for loader, loader_name in zip([train_loader, test_loader], ['train','test']):
                embeddings, predictions, ys, smiles_list = predict(loader, model)
                dfe = pd.DataFrame(embeddings)
                dfe['pred'] = predictions
                dfe['smiles'] = smiles_list
                dfe['y'] = ys
                dfe['group'] = loader_name
                dfe['with_aca'] = aca_used
                p_res.append(dfe)
        
        dfe = pd.concat(p_res)
        dfe.to_csv(os.path.join(result_save_dir,'embedding_prediction.pkl'))
        
        ############################################################################
        
        lii_fp = dfe.groupby(['group',
                     'with_aca']).apply(lambda x:get_structure_label_coherence(x.smiles.values, 
                                                                               x.y.values, 
                                                                               k = 10))
        
        lii_fp = lii_fp.loc[[( 'test', 'wo_aca'), ('train', 'wo_aca')]]
        lii_fx = dfe.groupby(['group',
                     'with_aca']).apply(lambda x:label_incoherence_index(x[range(512)].values, 
                                                                               x.y.values, 
                                                                               k = 10))
        
    
    
        dffp = lii_fp.to_frame('lii').reset_index()
        dffx = lii_fx.to_frame('lii').reset_index()
        
        dffp['repre'] = 'structure'
        dffx['repre'] = 'latent'
    
        dffx['method'] = dffx.repre + ':' + dffx.with_aca.astype(str)
        dffp['method'] = dffp.repre
    
    
        
        dflii = dffp._append(dffx)
        dflii['dataset'] = dataset_name
       
        dflii.to_csv(os.path.join(result_save_dir,'lii.csv'))
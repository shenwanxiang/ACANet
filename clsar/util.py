import os
import numpy as np

from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='SSD',
                 xd=None, t=None, transform=None,
                 pre_transform=None, smile_graph=None):
        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default =
        self.dataset = dataset
        print(self.processed_paths[0])
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, t, smile_graph)
            self.data = torch.load(self.processed_paths[0])
            #self.data = torch.load('C:\\Users\\CC\\Desktop\ceshi\\data\\processed\\ChEMBL_train.pt')


    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES
    # Y: list of labels
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, t,smile_graph):
        assert (len(xd) == len(t)), "The xd and y must be the same length!"
        data_list = []
        data_dict= {}
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            label = t[i]
            # 两种特殊情况。。
            #if type(smiles) == float:
            #    break;
            #if edge_index == []:
            #    edge_index = [[]]
            c_size, features, edge_index = smile_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(0, 1)
                                )
            GCNData.target = torch.LongTensor([label])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            #print(GCNData)
            if label in data_dict:
                data_dict[label].append(GCNData)
            else:
                data_dict[label] = []
                data_dict[label].append(GCNData)
            # append graph, label sequence to data list
            # data_list.append(GCNData)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        torch.save(data_dict, self.processed_paths[0])

    def rmse(y,f):
        rmse = sqrt(((y - f)**2).mean(axis=0))
        return rmse
    def mse(y,f):
        mse = ((y - f)**2).mean(axis=0)
        return mse
    def pearson(y,f):
        rp = np.corrcoef(y, f)[0,1]
        return rp
    def spearman(y,f):
        rs = stats.spearmanr(y, f)[0]
        return rs
    def ci(y,f):
        ind = np.argsort(y)
        y = y[ind]
        f = f[ind]
        i = len(y)-1
        j = i-1
        z = 0.0
        S = 0.0
        while i > 0:
            while j >= 0:
                if y[i] > y[j]:
                    z = z+1
                    u = f[i] - f[j]
                    if u > 0:
                        S = S + 1
                    elif u == 0:
                        S = S + 0.5
                j = j - 1
            i = i - 1
            j = i-1
        ci = S/z
        return ci
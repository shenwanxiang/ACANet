import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn

from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader as GDL
from torch.utils.data  import DataLoader as DL
from util import *
from itertools import combinations, permutations

# training
def predicting(model, device, test_data):
    model.eval()
    pass
def train(model, device, tripledata, optimizer, epoch):
    print('Begin training on {} samples...'.format(len(tripledata)))
    model.train()
    for idx, data in enumerate(tripledata):
        #data = data.to(device)  list不能放到GPU？
        print(data[0], type(data[0]))
        optimizer.zero_grad()
        output0 = model(data[0])
        output1 = model(data[1])
        output2 = model(data[2])
        #loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss = loss_fn(output0, output1, output2).float().to(device)
        loss.backward()
        optimizer.step()
        if idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           idx ,
                                                                           len(tripledata),
                                                                           100. * idx / len(tripledata),
                                                                           loss.item()))

datasets = [['SSD', 'mix'][int(sys.argv[1])]]
modeling = [GATNet][int(sys.argv[2])]
opt = [['train', 'test'][int(sys.argv[3])]]

print('opt:',opt)
#modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 5
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 2
NUM_EPOCHS = 100

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_train)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train').data
        print(train_data, type(train_data))
        kind = list(train_data.keys())
        print(kind)
        kindnum = len(kind)
        pair = []
        tripledata = []


        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.TripletMarginLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'

        print('opt:',opt,type(opt))
        print(processed_data_file_test)
        print(os.path.isfile(processed_data_file_test))
        #sys.exit()
        #opt是1就测试
        if opt[0] == 'test':
            if os.path.isfile(processed_data_file_test):
                    model.load_state_dict(torch.load(model_file_name))
                    model.eval()
                    test_data = torch.load(processed_data_file_test)
                    train_tmp = train_data.values()
                    train_drugs = []

                    datadict = []
                    for label, drugs in train_data.items():
                        for drug in drugs:
                            drug = model(drug)
                            print(drug)
                            datadict.append([drug, label])
                    print(datadict)

                    mindis = float('inf')
                    correct_num = 0
                    num = 0

                    for label, drugs in test_data.items():
                        for drug in drugs:
                            num += 1
                            emb = model(drug)
                            emb = emb.detach().numpy()
                            print('predict embedding:', emb)
                            for data in datadict:
                                comp_emb = data[0]
                                comp_label = data[1]
                                #emb = emb.detach().numpy()
                                comp_emb = comp_emb.detach().numpy()
                                #欧式距离
                                dis = np.linalg.norm(emb-comp_emb)
                                if mindis>dis:
                                    mindis = dis
                                    predict_label = comp_label
                            if predict_label == label:
                                correct_num += 1
                            print('predict_embedding:{}   predict_label:{}  label:{}'.format(emb, predict_label, label))
                    print('precision:', correct_num/num)
                    #sys.exit()
        #采样策略，找出对比学习的pair
        for i in range(kindnum):
            tmp1 = []
            num1 = len(train_data[kind[i]])
            if num1 < 2:
                continue
            else:
                tmp1.append(kind[i])
            for j in range(i+1,kindnum):
                tmp2 = tmp1
                num2 = len(train_data[kind[j]])
                if num2 < 2:
                    continue
                else:
                    tmp2.append(kind[j])
                    pair.append(tmp2)
        print(pair)
        for p in pair:
            anchor_pos = list(combinations(train_data[p[0]], 2))
            neg = list(combinations(train_data[p[1]], 1))
            for i in anchor_pos:
                for j in neg:
                    tmp = list(i+j)
                    tripledata.append(tmp)
        print('Number of pairs:', len(tripledata))

        print('TRIPLEDATA:', tripledata[0][0])

        # training the model
        for epoch in range(NUM_EPOCHS):
            train(model, device, tripledata, optimizer, epoch+1)
        torch.save(model.state_dict(), model_file_name)


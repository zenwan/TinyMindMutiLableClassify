#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

'base class for train'

__author__ = 'Ma Cong'

import torch
import torch.optim as optim
import numpy as np
from datetime import datetime

import checkpoint as cp
import load_data
import utils


class BaseTrain:
    def __init__(self, net, size=(128, 128), epoch_count=10, batch_size=256, use_gpu=True):
        self.cuda_is_ok = use_gpu
        self.cuda = torch.device("cuda" if self.cuda_is_ok else "cpu")
        self.img_size = size
        self.n_epoch = epoch_count
        self.batch_size = batch_size
        self.model = net.cuda() if self.cuda_is_ok else net
        self.criterion = torch.nn.MultiLabelSoftMarginLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        sets = load_data.Sets()
        trainset = sets.get_train_set()
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = sets.get_test_set()
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    def train(self, checkpoint_path):
        # 是否装载模型参数
        load = False

        if load:
            checkpoint = cp.load_checkpoint(address=checkpoint_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.n_epoch):
            self.train_one_epoch(epoch)

            # 保存参数
            checkpoint = {'epoch': epoch, 'state_dict': self.model.state_dict()}
            cp.save_checkpoint(checkpoint, address=checkpoint_path, index=epoch)

            self.test(epoch)

    def train_one_epoch(self, epoch):
        self.model.train()

        print(now())
        print('Begin training...')
        for batch_index, (datas, labels) in enumerate(self.trainloader, 0):
            datas = torch.tensor(datas, dtype=torch.float, device=self.cuda, requires_grad=False)
            datas = datas.view(-1, 3, self.img_size[0], self.img_size[1])
            #labels = labels.max(1)[1]
            labels = torch.tensor(labels, dtype=torch.float, device=self.cuda)
            self.optimizer.zero_grad()
            outputs = self.model(datas)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            if batch_index % 100 == 0:
                y_true = labels.cpu().numpy()
                y_pred = outputs.cpu().detach().numpy()
                y_pred = sigmoid(y_pred)
                #print_list(y_pred, labels, batch_index)
                predict = utils.precision(y_true, y_pred)
                recall = utils.recall(y_true, y_pred)
                fmeasure = utils.fmeasure(predict, recall)
                print('batch_index: [%d/%d]' % (batch_index, len(self.trainloader)),
                      'Train epoch: [%d]' % epoch,
                      'Loss:%.6f' % loss,
                      'Predict:%.6f' % predict,
                      'Recall:%.6f' % recall,
                      'F-measure:%.6f' % fmeasure)
                print(now())

    def test(self, epoch):
        self.model.eval()
        for batch_index, (datas, labels) in enumerate(self.testloader, 0):
            datas = torch.tensor(datas, dtype=torch.float, device=self.cuda, requires_grad=True)
            datas = datas.view(-1, 3, self.img_size[0], self.img_size[1])
            labels = torch.tensor(labels, dtype=torch.float, device=self.cuda)
            outputs = self.model(datas)

            if batch_index % 100 == 0:
                y_true = labels.cpu().numpy()
                y_pred = outputs.cpu().detach().numpy()
                y_pred = sigmoid(y_pred)
                predict = utils.precision(y_true, y_pred)
                recall = utils.recall(y_true, y_pred)
                fmeasure = utils.fmeasure(predict, recall)
                print('batch_index: [%d/%d]' % (batch_index, len(self.testloader)),
                      'Train epoch: [%d]' % epoch,
                      'Predict:%.6f' % predict,
                      'Recall:%.6f' % recall,
                      'F-measure:%.6f' % fmeasure)
                print(now())


def now():
    return datetime.now().strftime('%c')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def print_list(x, y, index):
    import pandas as pd

    d = {'x': x[0], 'y': y[0]}
    list = pd.DataFrame(data=d)
    list.to_csv('test%d.csv' % index)
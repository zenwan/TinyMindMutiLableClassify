#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'create models'

__author__ = 'Ma Cong'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


out_channel = 80
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, stride=2, padding=2),
            #torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
            #torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            #torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, out_channel, 3, stride=1, padding=1),
            #torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            #torch.nn.Dropout(p=0.5),
        )

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(14*14*out_channel, 6941),
            # torch.nn.BatchNorm1d(6941),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            # torch.nn.Linear(6941, 6941),
        )

    def forward(self, input):
        from torch.utils.checkpoint import checkpoint, checkpoint_sequential
        x = checkpoint_sequential(self.Conv, 3, input)
        x = x.view(-1, 14*14*out_channel)
        # x = checkpoint_sequential(self.Classes, 2, x)
        x = checkpoint(self.Classes, x)
        #x = self.Classes(x)
        return x


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(16, 32, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, out_channel, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(14*14*out_channel, 6941),
            # torch.nn.Sigmoid(),  # pytorch的multilabel_soft_margin_loss函数会进行sigmoid操作
            # nn.BatchNorm1d(6941),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            # torch.nn.Linear(6941, 6941),
        )

    def forward(self, input):
        from torch.utils.checkpoint import checkpoint, checkpoint_sequential
        x = checkpoint_sequential(self.Conv, 4, input)
        x = x.view(-1, 14*14*out_channel)
        x = checkpoint(self.Classes, x)
        return x

def net_res50():
    net = models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False

    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 12000),
        torch.nn.BatchNorm1d(12000),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(12000, 12000),
        torch.nn.BatchNorm1d(12000),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(12000, 6941),
    )
    return net

def net_vgg16():
    net = models.vgg16(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False

    num_ftrs = 25088
    net.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 6941),
            torch.nn.BatchNorm1d(6941),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(6941, 6941),
            torch.nn.BatchNorm1d(6941),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(6941, 6941),
        )
    return net
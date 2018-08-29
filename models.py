#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'create models'

__author__ = 'Ma Cong'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(14*14*128, 6941),
            nn.BatchNorm1d(6941),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(6941, 6941),
        )

    def forward(self, input):
        x = self.Conv(input)
        x = x.view(-1, 14*14*128)
        x = self.Classes(x)
        return x

def net_res50():
    net = models.resnet50(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False

    num_ftrs = net.fc.in_features
    net.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 6941),
            #torch.nn.Sigmoid(),
        )
    return net

def net_vgg16():
    net = models.vgg16(pretrained=True)
    for param in net.parameters():
        param.requires_grad = False

    num_ftrs = 25088
    net.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 6941),
            #torch.nn.Sigmoid(),
        )
    return net
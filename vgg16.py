#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'train by vgg16'

__author__ = 'Ma Cong'

import torch.optim as optim

from base_train import BaseTrain

class TrainVgg16(BaseTrain):
    def __init__(self, net, size=(128, 128), epoch_count=10, batch_size=256, use_gpu=True):
        super(TrainVgg16, self).__init__(net, size, epoch_count, batch_size, use_gpu)
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)

# class eval_vgg16(base_eval):
#     def __init__(self, size=(128,128), channel=1, test='test2\\', train='train\\'):
#         super(eval_vgg16, self).__init__(model_vgg16, size, channel, test, train)
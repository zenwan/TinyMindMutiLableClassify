#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' main module '

__author__ = 'Ma Cong'

import pandas as pd
import os
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

import load_data
import utils
import checkpoint as cp

class BaseEval():
    def __init__(self, net, size=(128, 128), use_gpu=False):
        self.cuda_is_ok = use_gpu
        self.cuda = torch.device("cuda" if self.cuda_is_ok else "cpu")
        self.img_size = size
        self.model = net

    def eval(self, checkpoint_path):
        self.model.eval()
        checkpoint = cp.load_checkpoint(address=checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])

        eval_set = load_data.Sets()
        eval_datas = eval_set.get_eval_set()
        eval_loader = torch.utils.data.DataLoader(eval_datas, batch_size=8, shuffle=True)
        pred_choice = []
        self.model.eval()
        for batch_index, datas in enumerate(eval_loader, 0):
            datas = torch.tensor(datas, dtype=torch.float, device=self.cuda, requires_grad=False)
            datas = datas.view(-1, 3, self.img_size[0], self.img_size[1])
            outputs = self.model(datas)
            outputs = outputs.cpu().detach().numpy()
            outputs = utils.sigmoid(outputs)
            outputs = np.round(np.clip(outputs, 0, 1))
            for out in outputs:
                index = []
                for i, v in enumerate(out):
                    if v == 1:
                        index.append(i)
                pred_choice.append(index)
            if batch_index % 100 == 0:
                print(batch_index)
            # pre_tags = eval_set.get_pre_tags(pred_choice)
            # import os
            # img_name = os.listdir('D:\\Datasets\\TinyMind图像标签竞赛预赛数据\\valid')
            # img_name = img_name[:8]
            # df = pd.DataFrame({'img_path': img_name, 'tags': pre_tags})
            # for i in range(df['tags'].shape[0]):
            #     df['tags'].iloc[i] = ','.join(str(e) for e in df['tags'].iloc[i])
            # df.to_csv('submit.csv', index=None)

        pre_tags = eval_set.get_pre_tags(pred_choice)
        import os
        img_name = os.listdir('D:\\Datasets\\TinyMind图像标签竞赛预赛数据\\valid')
        df = pd.DataFrame({'img_path': img_name, 'tags': pre_tags})
        for i in range(df['tags'].shape[0]):
            df['tags'].iloc[i] = ','.join(str(e) for e in df['tags'].iloc[i])
        df.to_csv('submit.csv', index=None)

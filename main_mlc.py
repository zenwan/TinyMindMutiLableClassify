#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' main module '

__author__ = 'Ma Cong'

import argparse
import os
import time

from torch.backends import cudnn
import cv2

import load_data
import models
from base_train import BaseTrain
from res50 import TrainRes50
from vgg16 import TrainVgg16

def main():
    '''main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', default='train', help='train or eval or data')
    parser.add_argument('--batch', default=4, type=int, help='batch size')
    parser.add_argument('--epoch', default=10, type=int, help='epoch count')
    parser.add_argument('--checkpoint', default='checkpoints\\', help='模型存储位置')
    parser.add_argument('--dropout', default=0.5, help='drop的概率')
    parser.add_argument('--imgwidth', default=224, type=int, help='图片预处理后的宽度')  # vgg16 input=[224, 224, 3] convout=[7, 7, 512]
    parser.add_argument('--imgheight', default=224, type=int, help='图片预处理后的高度')
    parser.add_argument('--loadcheckpoints', default=False, help='加载参数模型')
    parser.add_argument('--use_gpu', default=False, help='使用GPU')
    opt = parser.parse_args()
    print(opt)

    if opt.use_gpu:
        cudnn.enabled = True

    if opt.state == 'train':
        #train = BaseTrain(models.Net(), (opt.imgwidth, opt.imgheight), opt.epoch, opt.batch, opt.use_gpu)
        train = TrainRes50(models.net_res50(), (opt.imgwidth, opt.imgheight), opt.epoch, opt.batch, opt.use_gpu)
        #train = TrainVgg16(models.net_vgg16(), (opt.imgwidth, opt.imgheight), opt.epoch, opt.batch, opt.use_gpu)
        train.train(opt.checkpoint)

    elif opt.state == 'eval':
        import base_eval
        eval = base_eval.BaseEval(models.net_vgg16(), (opt.imgwidth, opt.imgheight))
        eval.eval(opt.checkpoint)

    elif opt.state == 'data':
        ld = load_data.LoadData()
        ld.load_data()
        ld.show_images()

    else:
        print('Error state, must choose from train and eval!')


if __name__ == '__main__':
    main()

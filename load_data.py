#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'picture preprocess'

import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

class LoadData():
    def __init__(self,
                 size=(224, 224),
                 path='D:\\Datasets\\TinyMind图像标签竞赛预赛数据\\'):
        self.root_path = path
        self.img_size = size

    def get_train_data(self):
        images = []
        for file in tqdm(self.file_train):
            image = self.get_image(file)
            images.append(image)

        return images, self.label_train

    def get_eval_data(self):
        images = []
        for file in tqdm(self.file_eval):
            image = self.get_image(file)
            images.append(image)

        return images, self.label_eval

    def show_images(self):
        images, _ = self.get_eval_data()
        fig, axes = plt.subplots(6, 6, figsize=(20, 20))

        j = 0
        for i, img in enumerate(images[:36]):
            axes[i // 6, j % 6].imshow(img)
            j += 1
        plt.show()

    def load_data(self):
        labels = self.load_tag_train()
        files = self.load_train_csv()
        hash_tag = self.hash_tag()
        index = np.arange(0, len(files), 1, dtype=np.int)
        np.random.seed(123)
        np.random.shuffle(index)
        index_eval = index[:int(len(files) * 0.99)]
        index_train = index[int(len(files) * 0.99):]
        self.file_train = files[index_train]
        self.label_train = labels[index_train]
        self.file_eval = files[index_eval]
        self.label_eval = labels[index_eval]
        np.random.seed()

    def get_files(self, folder_name):
        path = self.root_path + folder_name + '\\'
        files = os.listdir(path)
        filenames = []
        for file in files:
            file = path + file
            filenames.append(file)
        return np.array(filenames)

    def load_tag_train(self):
        tags = np.load(self.root_path + 'tag_train.npz')
        return np.array(tags['tag_train'])

    def load_train_csv(self):
        f = open(self.root_path + 'visual_china_train1.csv', encoding='utf-8')
        train_df = pd.read_csv(f)
        full_path_names = []
        files = train_df.iloc[:, 0]
        for file in files:
            full_path_names.append(self.root_path + 'train\\' + file)
        return np.array(full_path_names)

    def hash_tag(self):
        fo = open(self.root_path + 'valid_tags.txt', "r", encoding='utf-8')
        hash_tag = {}
        i = 0
        for line in fo.readlines():  # 依次读取每行
            line = line.strip()  # 去掉每行头尾空白
            hash_tag[i] = line
            i += 1
        return hash_tag

    def get_image(self, file):
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.asarray(img)
        img = cv2.resize(img, self.img_size)
        return img


class ImageSet(data.Dataset):
    def __init__(self, data):
        self.datas, self.labels = data

    def __getitem__(self, index):
        return torch.from_numpy(self.datas[index]), \
               torch.from_numpy(self.labels[index])

    def __len__(self):
        return len(self.datas)


class Sets:
    def __init__(self):
        self.ld = LoadData()
        self.ld.load_data()

    def get_train_set(self):
        return ImageSet(self.ld.get_train_data())

    def get_eval_set(self):
        return ImageSet(self.ld.get_eval_data())

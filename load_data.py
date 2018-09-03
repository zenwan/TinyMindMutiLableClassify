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
        print('Train data loaded')
        return images, self.label_train

    def get_test_data(self):
        images = []
        for file in tqdm(self.file_test):
            image = self.get_image(file)
            images.append(image)
        print('Test data loaded')
        return images, self.label_test

    def get_eval_data(self):
        images = []
        path_bin = self.root_path + 'binary_data\\eval_data.npy'
        if os.path.exists(path_bin):
            images = np.load(path_bin)
        else:
            files = self.get_files('valid\\')
            for file in tqdm(files):
                image = self.get_image(file)
                images.append(image)
            datas = np.array(images, dtype=np.uint8)
            np.save(path_bin, datas)

        print('Evalution datas loaded')
        return images

    def show_images(self):
        images, _ = self.get_test_data()
        fig, axes = plt.subplots(6, 6, figsize=(20, 20))

        j = 0
        for i, img in enumerate(images[:36]):
            axes[i // 6, j % 6].imshow(img)
            j += 1
        plt.show()

    def load_data(self):
        labels = self.load_tag_train()
        files = self.load_train_csv()
        index = np.arange(0, len(files), 1, dtype=np.int)
        np.random.seed(123)
        np.random.shuffle(index)
        index_test = index[:int(len(files) * 0.1)]
        index_train = index[int(len(files) * 0.1):]
        self.file_train = files[index_train]
        self.label_train = labels[index_train]
        self.file_test = files[index_test]
        self.label_test = labels[index_test]
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
        # print(hash_tag[2608])
        # print(hash_tag[3037])
        # print(hash_tag[3782])
        # print(hash_tag[550])
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


class EvalSet(data.Dataset):
    def __init__(self, data=None):
        self.datas = data

    def __getitem__(self, index):
        return torch.from_numpy(self.datas[index])

    def __len__(self):
        return len(self.datas)


class Sets:
    def __init__(self):
        self.ld = LoadData()
        self.ld.load_data()

    def get_train_set(self):
        return ImageSet(self.ld.get_train_data())

    def get_test_set(self):
        return ImageSet(self.ld.get_test_data())

    def get_eval_set(self):
        return EvalSet(self.ld.get_eval_data())

    def get_pre_tags(self, index):
        tags = self.ld.hash_tag()
        pre_tags = []
        for each_image in index:
            each_image_tags = []
            for i in each_image:
                each_image_tags.append(tags[i])
            pre_tags.append(each_image_tags)
        return pre_tags

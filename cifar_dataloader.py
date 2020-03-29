#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/29 3:58 下午
# @Author  : RuisongZhou
# @Mail    : rhyszhou99@gmail.com
import sys, os
import jittor.transform as trans
from jittor.dataset.dataset import Dataset
import pickle
import numpy as np
from PIL import Image

class cifar_dataset(Dataset):
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    def __init__(self, data_root="./data/cifar/", is_train=True, transform=None, target_transform=None, ):
        super(cifar_dataset,self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        self.root = data_root
        base_folder = 'cifar-10-batches-py'
        meta = {
            'filename': 'batches.meta',
            'key': 'label_names',
            'md5': '5ff9c542aee3614f3951f8cda6e48888',
        }
        if self.is_train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        self.data = []
        self.targets = []
        # now load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        path = os.path.join(self.root, base_folder, meta['filename'])
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        self.total_len = len(self.data)
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return trans.to_tensor(img), target


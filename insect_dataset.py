#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
from pathlib import Path
import numpy as np
import torch.utils.data as data


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.choice(24) / 24
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class InsectDataset(data.Dataset):
    def __init__(self, root, 
            num_points=2048, split='train', load_name=False, load_path=False,
            random_rotate=False, random_jitter=False, random_translate=False,
            classes=None, use_classes=None):
        """
        Args:
            root (str): Path of the parent directory of the dataset
            classes (_type_, optional): Class list ordered by id, beginning at 0. Defaults to CLASSES_6B.
            use_classes (_type_, optional): Load samples of only these classes. Defaults to None = all classes.
        """

        # assert num_points <= 2048 # why?

        self.root = Path(root)
        self.dataset_name = self.root.name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_path = load_path
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.classes = classes
        self.use_classes = use_classes
        
        # <class_name>:<class_id>
        self.class_id_map = dict(zip(classes, range(len(classes))))
        # <class_id>:<class_name>
        self.id_class_map = {v:k for k,v in self.class_id_map.items()}

        self.samples = []
        skipped_count = 0
        for f in self.root.glob("*/*.csv"):
            clas = f.parent.name
            if use_classes is not None and clas not in use_classes:
                # skip this sample if class is not used
                skipped_count += 1
                continue
            cla_id = self.class_id_map[clas]
            point_set = np.loadtxt(f, delimiter=',', skiprows=1, max_rows=self.num_points, usecols=(0,1,2)).astype(np.float32)
            rel_path = str(Path(clas) / f.name)
            self.samples.append( (point_set, cla_id, rel_path) )
        print(f"Loaded dataset from {self.root}; {skipped_count} samples skipped (not in use_classes)!")

    def __getitem__(self, index):
        sample = self.samples[index]
        point_set = sample[0]
        class_id = sample[1]
        class_name = self.id_class_map[class_id]
        rel_path = sample[2]

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([class_id]).astype(np.int64))
        # label = label.squeeze(0)
        
        if self.load_name:
            return point_set, label, class_name, rel_path
        else:
            return point_set, label

    def __len__(self):
        return len(self.samples)
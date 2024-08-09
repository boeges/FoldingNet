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

# make key (scene_id, instance_id, frag_index).
# example: "dragonfly/dragonfly_mu2-3_6_5.csv" becomes "mu2-3_6_5".
def frag_filename_to_id(fn):
    return "_".join(fn.replace(".csv","").split("_")[-3:])


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
    # A uses old order; B uses new order
    CLASSES_4A = ["bee","butterfly","dragonfly","wasp"]
    CLASSES_5A = ["bee","butterfly","dragonfly","wasp","insect"]
    CLASSES_6A = ["bee","butterfly","dragonfly","wasp","insect","other"]
    CLASSES_6B = ["other","insect","bee","butterfly","dragonfly","wasp"]
    CLASSES_7A = ["bee","butterfly","dragonfly","wasp","other","insect","bumblebee"]
    CLASSES_7B = ["other","insect","bee","butterfly","dragonfly","wasp","bumblebee"]


    def __init__(self, root, 
            num_points=2048, split='train', split_file=None, load_name=False, load_path=False,
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
        self.load_name = load_name
        self.load_path = load_path
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.classes = classes
        self.use_classes = use_classes
        self.split = split
        self.split_file = split_file

        # classes
        self.classes = InsectDataset.get_class_list(self.classes)
        if self.use_classes is None:
            self.use_classes = self.classes
        self.use_classes = InsectDataset.get_class_list(self.use_classes)

        # <class_name>:<class_id>
        self.class_id_map = dict(zip(self.classes, range(len(self.classes))))
        # <class_id>:<class_name>
        self.id_class_map = {v:k for k,v in self.class_id_map.items()}

        fids = None
        if self.split_file is not None and self.split_file != "all":
            assert self.split in ["train", "test"]
            assert self.split_file is not None

            # Use predefined split for train and test samples; Read sample ids from files (one for train and test)
            # File must be in the dataset directory!
            split_file_path = Path(self.root) / split_file
            print("Using train/test split file:", split_file_path)

            with open(split_file_path) as f:
                # row example: train,bee,hn-bee-1_0_17
                # fid example: hn-bee-1_0_17
                lines = f.read().splitlines()
                fids = [line.split(",")[-1] for line in lines if line.split(",")[0]==self.split]

            # print("Number of samples in", self.split, "file:", len(fids))

        self.samples = []
        skipped_count = 0
        for f in self.root.glob("*/*.csv"):
            clas = f.parent.name
            if self.use_classes is not None and clas not in self.use_classes:
                # skip this sample if class is not used
                skipped_count += 1
                continue
            if fids is not None:
                fid = frag_filename_to_id(f.name)
                if fid not in fids:
                    # skip this sample if sample is not in split
                    skipped_count += 1
                    continue
            cla_id = self.class_id_map[clas]
            point_set = np.loadtxt(f, delimiter=',', skiprows=1, max_rows=self.num_points, usecols=(0,1,2)).astype(np.float32)
            rel_path = str(Path(clas) / f.name)
            self.samples.append( (point_set, cla_id, rel_path) )
        print(f"Loaded dataset from {self.root}; {len(self.samples)} loaded; {skipped_count} skipped (not in use_classes or split file)")

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
    
    @staticmethod
    def get_class_list(args_classes):
        if args_classes=="4A":
            classes = InsectDataset.CLASSES_4A
        elif args_classes=="5A":
            classes = InsectDataset.CLASSES_5A
        elif args_classes=="6A":
            classes = InsectDataset.CLASSES_6A
        elif args_classes=="6B":
            classes = InsectDataset.CLASSES_6B
        elif args_classes=="7A":
            classes = InsectDataset.CLASSES_7A
        elif args_classes=="7B":
            classes = InsectDataset.CLASSES_7B
        elif isinstance(args_classes, str) and "," in args_classes:
            classes = args_classes.lower().split(",")
        elif isinstance(args_classes, list):
            classes = args_classes
        else:
            raise RuntimeError("Unsupported classes: " + str(args_classes))
        return classes
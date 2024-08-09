#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import torch.utils.data as data

from original_dataset import OriginalDataset
from insect_dataset import InsectDataset


class Dataset(data.Dataset):
    """
    Superclass for both dataset loaders.
    Loads either InsectDataset or OriginalDataset.
    """

    def __init__(self, root, dataset_name='insect', 
            num_points=4096, split='train', split_file=None, load_name=False,
            classes="6B", use_classes="6B",
            random_rotate=False, random_jitter=False, random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 
            'shapenetpart', 'modelnet10', 'modelnet40', 
            "insect"]

        if dataset_name.lower() == "insect":
            self._dataset = InsectDataset(root, 
                    num_points=num_points, split=split, split_file=split_file, load_name=load_name,
                    random_rotate=random_rotate, random_jitter=random_jitter, random_translate=random_translate,
                    classes=classes, use_classes=use_classes)
        else:
            self._dataset = OriginalDataset(root, dataset_name=dataset_name, 
                    num_points=num_points, split=split, load_name=load_name,
                    random_rotate=random_rotate, random_jitter=random_jitter, random_translate=random_translate)

    def __getitem__(self, item):
        return self._dataset.__getitem__(item)

    def __len__(self):
        return self._dataset.__len__()
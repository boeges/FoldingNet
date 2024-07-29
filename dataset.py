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

    def __init__(self, root, dataset_name='modelnet40', 
            num_points=2048, split='train', load_name=False,
            random_rotate=False, random_jitter=False, random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 
            'shapenetpart', 'modelnet10', 'modelnet40', 
            "insect"]

        if dataset_name.lower() == "insect":
            self._dataset = InsectDataset(self, root, dataset_name=dataset_name, 
                    num_points=num_points, split=split, load_name=load_name,
                    random_rotate=random_rotate, random_jitter=random_jitter, random_translate=random_translate)
        else:
            self._dataset = OriginalDataset(self, root, dataset_name=dataset_name, 
                    num_points=num_points, split=split, load_name=load_name,
                    random_rotate=random_rotate, random_jitter=random_jitter, random_translate=random_translate)

    def __getitem__(self, item):
        return self._dataset.__getitem__(item)

    def __len__(self):
        return self._dataset.__len__()
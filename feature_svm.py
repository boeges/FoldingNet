#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: inference.py
@Time: 2020/1/2 10:26 AM

Used to train an SVM on extract feature vectors and evalutate classifiaction performance.
"""

import os
import sys
import time
import shutil
import torch
import numpy as np
import h5py
import datetime
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.svm import LinearSVC
import bee_utils as bee
# from torchview import draw_graph

from model import ReconstructionNet, ClassificationNet
from dataset import Dataset
from utils import Logger


class FeatureSVM(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.no_cuda = args.no_cuda
        self.task = args.task
        self.split_file = args.split_file

        # Dir
        self.model_path = args.model_path
        self.snapshot_dir = Path(self.model_path).parent.parent
        self.feature_dir = self.snapshot_dir / "features"
        os.makedirs(self.feature_dir, exist_ok=True)

        # print args
        print(str(args))

        # get gpu id
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[0]

        # # generate dataset
        # self.infer_dataset_train = Dataset(
        #     root=args.dataset_root,
        #     dataset_name=args.dataset,
        #     split='train',
        #     split_file=self.split_file,
        #     num_points=args.num_points,
        #     classes=args.classes,
        #     use_classes=args.use_classes
        # )
        # self.infer_dataset_test = Dataset(
        #     root=args.dataset_root,
        #     dataset_name=args.dataset,
        #     split='test',
        #     split_file=self.split_file,
        #     num_points=args.num_points,
        #     classes=args.classes,
        #     use_classes=args.use_classes
        # )
        # print("Inference set size (train):", self.infer_dataset_train.__len__())
        # print("Inference set size (test):", self.infer_dataset_test.__len__())

        # # shape of one data sample: (2048, 3)
        # # example: [[-0.09796134  0.5089857   0.24515529], ...]
        # # shape of one label: (1,)
        # # example: [14]

        # train_data = []
        # train_label = []
        # for sample in self.infer_dataset_train:
        #     points = sample[0].numpy()
        #     label = sample[1].numpy()[0]
        #     train_data.append(points)
        #     train_label.append(label)
        #     # print(points, label)

        # test_data = []
        # test_label = []
        # for sample in self.infer_dataset_test:
        #     points = sample[0].numpy()
        #     label = sample[1].numpy()[0]
        #     test_data.append(points)
        #     test_label.append(label)
        #     # print(points, label)


        # load activations file
        df = pd.read_csv(args.activations_file, sep=",", header="infer")
        self.act_cols = df.columns[df.columns.str.startswith("act_")]

        df["sample_id"] = df["sample_path"].apply(lambda sample_path: bee.frag_filename_to_id_str(sample_path))

        # find out split ("train"/"test") of each fragment
        # fid example: "hn-dra-1_16_6"
        train_fids, test_fids = bee.read_split_file(args.split_file)
        df["split"] = df["sample_id"].apply(lambda sample_id: self.apply_split(sample_id, train_fids, test_fids))

        self.df = df

        print(self.df.head())


    def run(self):
        train_df = self.df[self.df["split"]=="train"].reset_index(drop=True)
        train_features = train_df.loc[:,self.act_cols]
        train_labels = train_df.loc[:,"target_index"]

        test_df = self.df[self.df["split"]=="test"].reset_index(drop=True)
        test_features = test_df.loc[:,self.act_cols]
        test_labels = test_df.loc[:,"target_index"]

        clf = LinearSVC(random_state=0, dual="auto")
        clf.fit(train_features, train_labels)

        result = clf.predict(test_features)
        result_series = pd.Series(result)

        result_df = test_df[["target_index","target_name","sample_id"]]
        result_df["pred_index"] = result_series
        result_df["correct"] = result_df["pred_index"] == result_df["target_index"]
        print(result_df.head())

        grouped_result_df = result_df.groupby("target_name").agg({"sample_id":"count", "correct":"sum"})
        grouped_result_df.rename({"sample_id":"samples"}, inplace=True, axis=1)
        grouped_result_df = grouped_result_df.reindex(["bee","bumblebee","wasp","butterfly","dragonfly","insect"])
        grouped_result_df["macc"] = grouped_result_df["correct"] / grouped_result_df["samples"]
        print(grouped_result_df.head())

        print("Latex:", " & ".join([f"{v:.3f}" for v in grouped_result_df["macc"].values]).replace(".",","))

        print("mAcc:", grouped_result_df["macc"].mean())

        sum_correct = (result_series == test_labels).sum()
        total_count = len(test_df.index)
        accuracy = sum_correct / total_count

        print("Transfer linear SVM accuracy: {:.2f}%".format(accuracy*100))


    def apply_split(self, sample_id, train_fids, test_fids):
        fid = sample_id
        if fid in train_fids:
            return "train"
        elif fid in test_fids:
            return "test"
        else:
            return None
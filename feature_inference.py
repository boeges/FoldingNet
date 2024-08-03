#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: inference.py
@Time: 2020/1/2 10:26 AM

Used to extract feature vectors.
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
# from torchview import draw_graph

from tensorboardX import SummaryWriter

from model import ReconstructionNet, ClassificationNet
from dataset import Dataset
from utils import Logger


class FeatureInference(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.no_cuda = args.no_cuda
        self.task = args.task

        # Dir
        self.snapshot_dir = Path(args.model_path).parent.parent
        self.feature_dir = self.snapshot_dir / "features"
        os.makedirs(self.feature_dir, exist_ok=True)

        # print args
        print(str(args))

        # get gpu id
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[0]

        # generate dataset
        self.infer_dataset_test = Dataset(
            root=args.dataset_root,
            dataset_name=args.dataset,
            split='test',
            num_points=args.num_points,
            load_name=True
        )
        self.infer_loader_test = torch.utils.data.DataLoader(
            self.infer_dataset_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers
        )
        print("Inference set size (test):", self.infer_loader_test.dataset.__len__())

        # shape of one data sample: (2048, 3)
        # example: [[-0.09796134  0.5089857   0.24515529], ...]
        # shape of one label: (1,)
        # example: [14]

        # initialize model
        self.model = ReconstructionNet(args)
        if args.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if not args.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = self.model.cuda(self.gpu_ids[0])
        
    def run(self):
        self.model.eval()
        
        activations_per_sample = []
        for iter, (pts, lbs, classs, paths) in enumerate(self.infer_loader_test):
            if not self.no_cuda:
                pts = pts.cuda(self.first_gpu)
                lbs = lbs.cuda(self.first_gpu)

            output, feature = self.model(pts)

            # vis
            # model_graph = draw_graph(self.model, input_data=pts, device='cuda', save_graph=True, filename="model", expand_nested=True)
            # saved_vis = True
            # exit()

            # for each sample add activations to a list
            for activations1, target1, clas1, path1 in zip(
                        feature.detach().cpu().numpy().squeeze(1), lbs.detach().cpu().numpy().squeeze(1), 
                        classs, paths):
                target_name = clas1
                target_index = target1
                activations_per_sample.append( [path1, target_index, target_name, *activations1] )

        print("Finish getting feature vectors. count =", len(activations_per_sample))

        # save features as pandas df
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        activations_path = Path(self.feature_dir) / f"activations_per_sample_{timestr}.csv"
        activations_header = ["act_"+str(i) for i in range(len(activations_per_sample[0]) - 3)] # Important: Subtract number of other columns!
        
        fragments_df = pd.DataFrame(activations_per_sample, \
                columns=["sample_path", "target_index", "target_name", *activations_header])
        fragments_df.to_csv(activations_path, index=False, header=True, decimal='.', sep=',', float_format='%.6f')
        print("Saved to:", activations_path)

        return self.feature_dir
    

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            if key[:10] == 'classifier':
                continue
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")

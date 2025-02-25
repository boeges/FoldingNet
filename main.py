#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main.py
@Time: 2020/1/2 10:26 AM
"""

import argparse

from reconstruction import Reconstruction
from classification import Classification
from inference import Inference
from feature_inference import FeatureInference
from feature_svm import FeatureSVM
from svm import SVM


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--task', type=str, default='reconstruct', metavar='N',
                        choices=['reconstruct', 'classify', "feature_inference", "feature_svm"],
                        help='Experiment task, [reconstruct, classify, feature_inference, feature_svm]')
    parser.add_argument('--encoder', type=str, default='foldingnet', metavar='N',
                        choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
                        help='Encoder to use, [foldingnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--feat_dims', type=int, default=512, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=None, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--shape', type=str, default='plane', metavar='N',
                        choices=['plane', 'sphere', 'gaussian'],
                        help='Shape of points to input decoder, [plane, sphere, gaussian]')
    parser.add_argument('--dataset', type=str, default='shapenetcorev2', metavar='N',
                        choices=['shapenetcorev2','modelnet40', 'modelnet10','insect'],
                        help='Encoder to use, [shapenetcorev2,modelnet40, modelnet10]')
    parser.add_argument('--use_rotate', action='store_true',
                        help='Rotate the pointcloud before training')
    parser.add_argument('--use_translate', action='store_true',
                        help='Translate the pointcloud before training')
    parser.add_argument('--use_jitter', action='store_true',
                        help='Jitter the pointcloud before training')
    parser.add_argument('--dataset_root', type=str, default='../dataset', help="Dataset root path")
    parser.add_argument('--gpu', type=str, help='Id of gpu device to be used', default='0')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N',
                        help='Save snapshot interval ')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Enables CUDA training')
    parser.add_argument('--get_activations', action='store_true',
                        help='Save the feature vector of a trained encoder')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Num of points to use')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to load model')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Num of output classes of the model for classification')
    parser.add_argument('--split_file', type=str, default='',
                        help='File name (or path) specifiing which samples are from the train and test split')
    parser.add_argument('--activations_file', type=str, default='',
                        help='File path to the file containing activations; Should be under features/ .')
    parser.add_argument('--classes', type=str, default="6B", 
                        help='Names of classes in order! Comma separated class names (e.g. bee,butterfly,...) or a predefined list [6A, 6B, ...] for default class list')
    parser.add_argument('--use_classes', type=str, default=None, 
                        help='Names of classes to load samples from. Comma separated class names (e.g. bee,butterfly,...) or a predefined list [6A, 6B, ...] for default class list')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()

    if args.task == 'reconstruct':
        reconstruction = Reconstruction(args)
        reconstruction.run()
    elif args.task == 'classify':
        classification = Classification(args)
        classification.run()
    elif args.task == "feature_inference":
        feature_inference = FeatureInference(args)
        feature_inference.run()
    elif args.task == "feature_svm":
        featureSvm = FeatureSVM(args)
        featureSvm.run()

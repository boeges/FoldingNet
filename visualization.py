#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: visualization.py
@Time: 2020/1/2 10:26 AM
"""

import os
import time
import numpy as np
import torch
import itertools
import argparse
from glob import glob
from pathlib import Path
import pandas as pd

from model import ReconstructionNet

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    return result

xml_head = \
"""
<scene version="0.5.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="ldrfilm">
            <integer name="width" value="1600"/>
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
            <scale value="0.7"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

xml_tail = \
"""
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def mitsuba(pcl, path, clr=None):
    xml_segments = [xml_head]

    # pcl = standardize_bbox(pcl, 2048)
    # pcl = pcl - np.expand_dims(np.mean(pcl, axis=0), 0)  # center
    # dist = np.max(np.sqrt(np.sum(pcl ** 2, axis=1)), 0)
    # pcl = pcl / dist  # scale

    pcl = pcl[:,[2,0,1]]
    pcl[:,0] *= -1
    h = np.min(pcl[:,2])

    if clr == "plane":
        clrgrid = [[0, 1, 45], [1, 0, 45]]
        b = np.linspace(*clrgrid[0])
        c = np.linspace(*clrgrid[1])
        color_all = np.array(list(itertools.product(b, c)))
        color_all = np.concatenate((np.linspace(1, 0, 2025)[..., np.newaxis], color_all), axis=1)
    elif clr == "sphere":
        color_all = np.load("sphere.npy")
        color_all = (color_all + 0.3) / 0.6
    elif clr == "gaussian":
        color_all = np.load("gaussian.npy")
        color_all = (color_all + 0.3) / 0.6

    for i in range(pcl.shape[0]):
        if clr == None:
            color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5)
        elif clr in ["plane", "sphere", "gaussian"]:
            color = color_all[i]
        else:
            color = clr
        if h < -0.25:
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2]-h-0.6875, *color))
        else:
            xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
    xml_segments.append(xml_tail)

    xml_content = str.join('', xml_segments)

    with open(path, 'w') as f:
        f.write(xml_content)

def load_pretrain(model, pretrain):
    state_dict = torch.load(pretrain, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module.'
        else:
            name = key
        new_state_dict[name] = val
    model.load_state_dict(new_state_dict)
    print(f"Load model from {pretrain}")
    return model   


def visualize(args):
    # create exp directory
    assert args.model_path.endswith(".pkl")

    model_path = Path(args.model_path)
    model_name = model_path.stem
    snapshot_root = model_path.parent.parent
    datestr = time.strftime('%Y-%m-%d_%H-%M-%S')
    save_root = snapshot_root / "mitsuba" / datestr
    save_root.mkdir(exist_ok=True, parents=True)
    
    # initialize dataset
    from dataset import Dataset
    dataset = Dataset(
        root=args.dataset_root,
        dataset_name=args.dataset,
        split='all',
        num_points=args.num_points,
        random_translate=False,
        random_rotate=False,
        random_jitter=False,
        classes="6B",
        load_name=True
    )

    pts = None
    class_name = None
    rel_path = None

    # find_path = "bee/hn-bee-1_9_9.csv"
    # find_path = "bee/mb-bum2-2_4_8.csv"
    # find_path = "bee/hn-depth-1_14_0.csv"
    find_path = "dragonfly/hn-dra-1_2_2.csv"
    found = False
    for i in range(len(dataset)):
        pts, label, class_name, rel_path = dataset[i]
        if rel_path == find_path:
            found = True
            break
    
    assert found
    
    # load data from dataset
    # pts, label, class_name, rel_path = dataset[args.item]
    sample_name = rel_path.split("/")[-1].replace(".csv", "")

    print(f"Dataset: {args.dataset}, split: {args.split}, item: {args.item}, category: {class_name}, sample: {sample_name}")

    # generate XML file for original point cloud
    if args.draw_original:
        # save_path = save_root / (args.dataset + '_' + args.split + str(args.item) + '_' + str(class_name) + '_origin.xml')
        # color = [0.4, 0.4, 0.6]
        # mitsuba(pts.numpy(), save_path, color)
        save_path = save_root / (sample_name + '_origin.csv')
        df = pd.DataFrame(pts, columns=["x","y","t"])
        df.to_csv(save_path, index=False, header=True, decimal='.', sep=',', float_format='%.3f')
        print("Saved original PC to", save_path)


    # generate XML file for decoder souce point 
    if args.draw_source_points:
        if args.shape == 'plane':
            assert args.num_points == 2048
            meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
            x = np.linspace(*meshgrid[0])
            y = np.linspace(*meshgrid[1])
            points = np.array(list(itertools.product(x, y)))
            points = np.concatenate((points,np.zeros(2025)[..., np.newaxis]), axis=1)
        elif args.shape == 'sphere':
            assert args.num_points == 2048
            points = np.load("sphere.npy")
        elif args.shape == 'gaussian':
            if args.num_points == 4096:
                points = np.load("gaussian4096.npy")
            else:
                points = np.load("gaussian.npy")
        # save_path = os.path.join(save_root, args.dataset + '_' + args.split + str(args.item) + '_' + str(class_name) + '_epoch0.xml')
        # mitsuba(points, save_path, clr=args.shape)
        save_path = save_root / (sample_name + '_epoch0.csv')
        df = pd.DataFrame(points, columns=["x","y","t"])
        df.to_csv(save_path, index=False, header=True, decimal='.', sep=',', float_format='%.3f')
        print("Saved epoch0 PC to", save_path)
        

    # initialize model
    model = ReconstructionNet(args)

    model = load_pretrain(model, model_path)
    model.eval()

    pts =  np.expand_dims(pts,0)
    pts = torch.from_numpy(pts)
    reconstructed_pl, _ = model(pts) # .view(1, 2048, 3)
    reconstructed_pl = reconstructed_pl.detach().numpy()
    reconstructed_pl = np.squeeze(reconstructed_pl, axis=0)

    # save_path = os.path.join(save_root, file[-1][:-4] + args.split + str(args.item) + '_' + str(class_name) + '.xml')
    # mitsuba(reconstructed_pl[0].detach().numpy(), save_path, clr=args.shape)

    save_path = save_root / (sample_name + "reconstruct_" + model_name + '.csv')
    df = pd.DataFrame(reconstructed_pl, columns=["x","y","t"])
    df.to_csv(save_path, index=False, header=True, decimal='.', sep=',', float_format='%.3f')
    print("Saved reconstruction PC to", save_path)
    

if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--exp_root', type=str, default=None, metavar='N',
                        help='Path of the experiment. (eg ./snapshot/abc)')
    parser.add_argument('--item', type=int, default=0, metavar='N',
                        help='Item of point cloud to load')
    parser.add_argument('--split', type=str, default='train', metavar='N',
                        choices=['train','test', 'val', 'trainval', 'all'],
                        help='Split to use, [foldingnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--encoder', type=str, default='foldingnet', metavar='N',
                        choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
                        help='Encoder to use, [foldingnet, dgcnn_cls, dgcnn_seg]')
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
    parser.add_argument('--dataset_root', type=str, default='../dataset', help="Dataset root path")
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Path to load model')
    parser.add_argument('--draw_original', action='store_true',
                        help='Draw original point cloud')
    parser.add_argument('--draw_source_points', action='store_true',
                        help='Draw source points for decoder')
    args = parser.parse_args()

    print(str(args))

    visualize(args)
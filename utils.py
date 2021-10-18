# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 20:57
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import glob
import os
import torch
import torch.nn as nn
import shutil
import numpy as np
import matplotlib.pyplot as plt
import dataloaders.transforms as t
from PIL import Image

cmap = plt.cm.jet


def parse_command():
    modality_names = ['rgb', 'rgbd', 'd']

    import argparse
    parser = argparse.ArgumentParser(description='FCRN')
    parser.add_argument('--decoder', default='upproj', type=str)
    parser.add_argument('--resume',
                        default=None,
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: ./run/run_1/checkpoint-5.pth.tar)')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='mini-batch size (default: 4)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate (default 0.0001)')
    parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler. '
                                                                   'See documentation of ReduceLROnPlateau.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--dataset', type=str, default="nyu")
    parser.add_argument('--dataset_type', type=str, default="dataset0")
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    return args


def get_output_directory(args):
    if args.resume:
        return os.path.dirname(args.resume)
    else:
        save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        save_dir_root = os.path.join(save_dir_root, 'result', args.decoder)
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

        save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
        return save_dir


# 保存检查点
def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)


import numpy as np
import cv2
from pathlib import Path
from metrics import MetricComputation

def colored_depthmap(depth, d_min=None, d_max=None, do_mapping=True):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_relative *= 255
    depth_relative = depth_relative.astype(np.uint8)
    if do_mapping:return cv2.applyColorMap(depth_relative, cv2.COLORMAP_TURBO)  # H, W, C
    return depth_relative


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge

def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = cv2.imwrite(filename, img_merge.astype('uint8'))

def show_item(item):
    img, depth = item
    if img.ndim == 4:
        img = img.squeeze(0)
    img = 255 * np.transpose(img.cpu().numpy(), (1, 2, 0))  # H, W, C
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if depth.ndim == 4:
        depth = depth.squeeze(0).squeeze(0)
    elif depth.ndim == 3:
        depth = depth.squeeze(0)
    depth = depth.cpu().numpy()
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth = colored_depthmap(depth, d_min, d_max)
    cv2.imshow("item", np.hstack([img, depth]).astype('uint8'))
    cv2.waitKey(0)

def save_images(path, idx, rgb=None, depth_gt=None, depth_pred=None):
    if path is None:return
    
    path=Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path = path.as_posix()
    min_ = np.finfo(np.float16).max
    max_ = np.finfo(np.float16).min
    if not rgb is None:
        if rgb.ndim == 4: rgb = rgb.squeeze(0)
        rgb = 255 * np.transpose(rgb.cpu().numpy(), (1, 2, 0))  # H, W, C
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        save_image(rgb, "{}/{}_rgb.jpg".format(path, idx))
    if not depth_gt is None:
        if depth_gt.ndim == 4: depth_gt = depth_gt.squeeze(0)
        if depth_gt.ndim == 3: depth_gt = depth_gt.squeeze(0)
        depth_gt = depth_gt.cpu().numpy()
        min_, max_ = min(np.min(depth_gt), min_), max(np.max(depth_gt), max_)
        
    if not depth_pred is None:
        if depth_pred.ndim == 4: depth_pred = depth_pred.squeeze(0)
        if depth_pred.ndim == 3: depth_pred = depth_pred.squeeze(0)
        depth_pred = depth_pred.cpu().numpy()
        min_, max_ = min(np.min(depth_pred), min_), max(np.max(depth_pred), max_)
        
    if not depth_pred is None:
        depth_pred = colored_depthmap(depth_pred, min_, max_)
        save_image(depth_pred, "{}/{}_pred.jpg".format(path, idx))

    if not depth_gt is None:
        depth_gt = colored_depthmap(depth_gt, min_, max_)
        save_image(depth_gt, "{}/{}_gt.jpg".format(path, idx))
    
    

def show_pred(depth_pred):
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    d_min = np.min(depth_pred_cpu)
    d_max = np.max(depth_pred_cpu)
    depth_target_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    cv2.imshow("pred", depth_target_col.astype('uint8'))
    cv2.waitKey(0)


def get_depth_sid(args, labels, cuda = False):
    if args == 'kitti':
        min = 0.001
        max = 80.0
        K = 71.0
    elif args == 'nyu':
        min = 0.02
        max = 10.0
        K = 90.0
    elif args == 'floorplan3d':
        min = 0.0552
        max = 10.0
        K = 68.0
    elif args == 'Structured3D':
        min = 0.02
        max = 10.0
        K = 90.0
    else:
        print('No Dataset named as ', args.dataset)

    if cuda:
        alpha_ = torch.tensor(min).cuda()
        beta_ = torch.tensor(max).cuda()
        K_ = torch.tensor(K).cuda()
    else:
        alpha_ = torch.tensor(min)
        beta_ = torch.tensor(max)
        K_ = torch.tensor(K)

    # print('label size:', labels.size())
    if not alpha_ == 0.0:
        depth = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * labels / K_)
    else:
        depth = torch.exp(torch.log(beta_) * labels / K_)
    # depth = alpha_ * (beta_ / alpha_) ** (labels.float() / K_)
    # print(depth.size())
    return depth.float()


def get_labels_sid(args, depth, cuda=False):
    if args == 'kitti':
        alpha = 0.001
        beta = 80.0
        K = 71.0
    elif args == 'nyu':
        alpha = 0.02
        beta = 10.0
        K = 90.0
    elif args == 'floorplan3d':
        alpha = 0.0552
        beta = 10.0
        K = 68.0
    elif args == 'Structured3D':
        alpha = 0.02
        beta = 10.0
        K = 90.0
    else:
        print('No Dataset named as ', args.dataset)

    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    K = torch.tensor(K)

    if cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()
        K = K.cuda()
    if not alpha == 0.0:
        labels = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    else:
        labels = K * torch.log(depth) / torch.log(beta)
    if cuda:
        labels = labels.cuda()
    return labels.int()

def depth2label_sid(depth, K=100.0, alpha=0.02, beta=10.0, cuda=False):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    K = torch.tensor(K)

    if cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()
        K = K.cuda()

    label = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    if cuda:
        label = torch.max(label, (torch.zeros(label.shape).double()).cuda()) # prevent negative label.
        label = label.cuda()
    else:
        label = torch.max(label, torch.zeros(label.shape).double())
    return label.int()

def adjust_padding(t):
    #pads a 128x128 image to 226x226
    return nn.ZeroPad2d((49,49,49,49))(t)

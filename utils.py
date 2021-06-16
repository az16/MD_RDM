# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 20:57
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import glob
import os
import torch
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


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


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
        K = 68.0
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
        K = 68
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

def depth2label_sid(depth, K=90.0, alpha=0.02, beta=10.0, cuda=False):
    alpha = torch.tensor(alpha)
    beta = torch.tensor(beta)
    K = torch.tensor(K)

    if cuda:
        alpha = alpha.cuda()
        beta = beta.cuda()
        K = K.cuda()

    label = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    if cuda:
        label = torch.max(label, torch.zeros(label.shape).cuda()) # prevent negative label.
        label = label.cuda()
    else:
        label = torch.max(label, torch.zeros(label.shape))
    return label.int()



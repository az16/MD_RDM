import torch
import torch.nn as nn
import math
import numpy as np



class Ordinal_Loss():
    """
    Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
    over the entire image domain:
    """

    def __init__(self):
        self.loss = 0.0

    def calc(self, ord_labels, target):
        """
        :param ord_labels: ordinal labels for each position of Image I.
        :param target:     the ground_truth discreted using SID strategy.
        :return: ordinal loss
        """
        # assert pred.dim() == target.dim()
        # invalid_mask = target < 0
        # target[invalid_mask] = 0
        N, C, H, W = ord_labels.size()
        ord_num = C
        # print('ord_num = ', ord_num)

        self.loss = 0.0

        # faster version
        # if torch.cuda.is_available():
        #     K = torch.zeros((N, C, H, W), dtype=torch.int).cuda()
        #     for i in range(ord_num):
        #         K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int).cuda()
        # else:
        K = torch.zeros((N, C, H, W), dtype=torch.int)
        for i in range(ord_num):
            K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int)

        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()

        one = torch.ones(ord_labels[mask_1].size())
        # if torch.cuda.is_available():
        #     one = one.cuda()

        self.loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8))) \
                     + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8)))

        # del K
        # del one
        # del mask_0
        # del mask_1

        N = N * H * W
        self.loss /= (-N)  # negative
        return self.loss

class RMSE_Loss():
    def __init__(self):
        self.loss = 0.0
    
    def calc(self, m1, m2):
        loss = torch.mean((m1-m2)**2)**0.5
        return loss

class L2_Loss():
    def __init__(self):
        self.loss = []

    def calc(self, yhat, y):
        sqr_err_list = []
        for i in range(7):
            sqr_err_list.append(torch.sum(torch.abs(y[i]-yhat[i])**2))
        loss = sqr_err_list
        return loss
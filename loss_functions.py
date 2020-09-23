"""
This code is base on the source code of
GradNet: Unsupervised Deep Screened Poisson Reconstruction for Gradient-Domain Rendering
Credit to Guo et al.
"""

import torch
import torch.nn.functional as F
import numpy as np

class LossFunction():
    def __init__(self, args):
        # self.mu = args.mu
        # self.c = args.c
        self.feature_channels = args.feature_channel

        self.select_right = np.array((0, 1), dtype=np.float32)
        self.select_right = self.select_right.reshape(1, 1, 1, 2)
        self.select_right = torch.from_numpy(self.select_right)

        self.select_bottom = np.array((0, 1), dtype=np.float32)
        self.select_bottom = self.select_bottom.reshape(1, 1, 2, 1)
        self.select_bottom = torch.from_numpy(self.select_bottom)

        self.select_left = np.array((1, 0), dtype=np.float32)
        self.select_left = self.select_left.reshape(1, 1, 1, 2)
        self.select_left = torch.from_numpy(self.select_left)

        self.select_top = np.array((1, 0), dtype=np.float32)
        self.select_top = self.select_top.reshape(1, 1, 2, 1)
        self.select_top = torch.from_numpy(self.select_top)

        self.select_right_7 = np.array((0, 1) * self.feature_channels, dtype=np.float32)
        self.select_right_7 = self.select_right_7.reshape(self.feature_channels, 1, 1, 2)
        self.select_right_7 = torch.from_numpy(self.select_right_7)

        self.select_bottom_7 = np.array((0, 1) * self.feature_channels, dtype=np.float32)
        self.select_bottom_7 = self.select_bottom_7.reshape(self.feature_channels, 1, 2, 1)
        self.select_bottom_7 = torch.from_numpy(self.select_bottom_7)

        self.select_left_7 = np.array((1, 0) * self.feature_channels, dtype=np.float32)
        self.select_left_7 = self.select_left_7.reshape(self.feature_channels, 1, 1, 2)
        self.select_left_7 = torch.from_numpy(self.select_left_7)

        self.select_top_7 = np.array((1, 0) * self.feature_channels, dtype=np.float32)
        self.select_top_7 = self.select_top_7.reshape(self.feature_channels, 1, 2, 1)
        self.select_top_7 = torch.from_numpy(self.select_top_7)

        self.get_dx = np.array((-1, 1), dtype=np.float32)
        self.get_dx = self.get_dx.reshape(1, 1, 1, 2)
        self.get_dx = torch.from_numpy(self.get_dx)

        self.get_dy = np.array((-1, 1), dtype=np.float32)
        self.get_dy = self.get_dy.reshape(1, 1, 2, 1)
        self.get_dy = torch.from_numpy(self.get_dy)

        self.get_feature_dx = np.array((-1, 1) * self.feature_channels, dtype=np.float32)
        self.get_feature_dx = self.get_feature_dx.reshape(self.feature_channels, 1, 1, 2)
        self.get_feature_dx = torch.from_numpy(self.get_feature_dx)

        self.get_feature_dy = np.array((-1, 1) * self.feature_channels, dtype=np.float32)
        self.get_feature_dy = self.get_feature_dy.reshape(self.feature_channels, 1, 2, 1)
        self.get_feature_dy = torch.from_numpy(self.get_feature_dy)

        self.get_feature_neg_dx = -1 * np.array((-1, 1) * self.feature_channels, dtype=np.float32)
        self.get_feature_neg_dx = self.get_feature_neg_dx.reshape(self.feature_channels, 1, 1, 2)
        self.get_feature_neg_dx = torch.from_numpy(self.get_feature_neg_dx)

        self.get_feature_neg_dy = -1 * np.array((-1, 1) * self.feature_channels, dtype=np.float32)
        self.get_feature_neg_dy = self.get_feature_neg_dy.reshape(self.feature_channels, 1, 2, 1)
        self.get_feature_neg_dy = torch.from_numpy(self.get_feature_neg_dy)

        if args.cuda:
            self.select_right = self.select_right.cuda()
            self.select_right_7 = self.select_right_7.cuda()
            self.select_bottom = self.select_bottom.cuda()
            self.select_bottom_7 = self.select_bottom_7.cuda()
            self.select_left = self.select_left.cuda()
            self.select_left_7 = self.select_left_7.cuda()
            self.select_top = self.select_top.cuda()
            self.select_top_7 = self.select_top_7.cuda()
            self.get_dx = self.get_dx.cuda()
            self.get_dy = self.get_dy.cuda()
            self.get_feature_dx = self.get_feature_dx.cuda()
            self.get_feature_neg_dx = self.get_feature_neg_dx.cuda()
            self.get_feature_dy = self.get_feature_dy.cuda()
            self.get_feature_neg_dy = self.get_feature_neg_dy.cuda()

    def eval_l1_loss(self, nI, I):
        return torch.mean(torch.abs(nI - I))

    def eval_grad_loss(self, nI, dx, dy):
        dx = F.conv2d(dx, self.select_left)
        dy = F.conv2d(dy, self.select_top)
        oIdx = F.conv2d(nI, self.get_dx)
        oIdy = F.conv2d(nI, self.get_dy)
        return torch.mean(torch.abs(oIdx - dx)) + torch.mean(torch.abs(oIdy - dy))

    def eval_feature_loss(self, nI, grad, feature, I):
        # calc the estimated grad
        Fdx = F.conv2d(feature, self.get_feature_dx, groups=self.feature_channels)
        gradx = F.conv2d(grad, self.select_left_7, groups=self.feature_channels)
        gradx = torch.sum(gradx * Fdx, 1)
        gradx = gradx.unsqueeze(1)
        Fdy = F.conv2d(feature, self.get_feature_dy, groups=self.feature_channels)
        grady = F.conv2d(grad, self.select_top_7, groups=self.feature_channels)
        grady = torch.sum(grady * Fdy, 1)
        grady = grady.unsqueeze(1)
        Fndx = F.conv2d(feature, self.get_feature_neg_dx, groups=self.feature_channels)
        ngradx = F.conv2d(grad, self.select_right_7, groups=self.feature_channels)
        ngradx = torch.sum(ngradx * Fndx, 1)
        ngradx = ngradx.unsqueeze(1)
        Fndy = F.conv2d(feature, self.get_feature_neg_dy, groups=self.feature_channels)
        ngrady = F.conv2d(grad, self.select_bottom_7, groups=self.feature_channels)
        ngrady = torch.sum(ngrady * Fndy, 1)
        ngrady = ngrady.unsqueeze(1)
        # calc weights
        weight_x = F.conv2d(I, self.get_dx, groups=1) ** 2
        weight_y = F.conv2d(I, self.get_dy, groups=1) ** 2
        weight_x = torch.exp(-weight_x * 9)
        weight_y = torch.exp(-weight_y * 9)
        # eval regularization term
        nI_left = F.conv2d(nI, self.select_left, groups=1)
        nI_right = F.conv2d(nI, self.select_right, groups=1)
        nI_top = F.conv2d(nI, self.select_top, groups=1)
        nI_bottom = F.conv2d(nI, self.select_bottom, groups=1)
        feature_loss = torch.mean(torch.abs(nI_left + gradx - nI_right) * weight_x) + torch.mean(
            torch.abs(nI_right + ngradx - nI_left) * weight_x)
        feature_loss += torch.mean(torch.abs(nI_top + grady - nI_bottom) * weight_y) + torch.mean(
            torch.abs(nI_bottom + ngrady - nI_top) * weight_y)
        return feature_loss





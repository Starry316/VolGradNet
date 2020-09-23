"""
Common basic modules

"""

import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Basis convolutional layer with leaky relu
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        ret = F.leaky_relu(self.conv(x))
        return ret


class DeConvBlock(nn.Module):
    """
    Deconvolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DeConvBlock, self).__init__()
        self.de_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        ret = F.relu(self.de_conv(x))
        return ret


class ResidualBlock(nn.Module):
    """
    ResidualBlock
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv_1 = ConvBlock(channels, channels, kernel_size, stride=1, padding=padding)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding)

    def forward(self, x):
        ret = x + self.conv_2(self.conv_1(x))
        return ret


class ProcessingBlock(nn.Module):
    """
    basic module for NGPT
    """
    def __init__(self, in_channels, out_channels, padding=0):
        super(ProcessingBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, 1, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x1))
        return x2
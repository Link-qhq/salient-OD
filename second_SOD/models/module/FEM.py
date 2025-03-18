# Feature Enhancement Module 特征增强

import torch
from torch import nn
from torch.nn import functional as F
from models.module.common import conv3x3_bn_relu


# context Enhancement Module
# Dual-path multi-branch feature residual network for salient object detection
class CEM(nn.Module):
    def __init__(self, in_ch, out_ch, kerSize=3):
        super(CEM, self).__init__()
        self.conv = conv3x3_bn_relu(in_ch, out_ch)
        self.conv1 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kerSize, 1, kerSize // 2), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kerSize, 1, kerSize // 2), nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, x):
        return self.conv2(self.conv1(self.conv(x)))

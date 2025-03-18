# Multi-scale feature fusion 多尺度特征提取或融合模块

import torch
from torch import nn
from torch.nn import functional as F
from models.module.common import conv3x3_bn_relu


# Multi-scale Feature Interaction Module (MFIM)
# Dual-path multi-branch feature residual network for salient object detection
class MFIM(nn.Module):
    def __init__(self, channel):
        super(MFIM, self).__init__()
        self.self_conv = nn.Conv2d(channel, channel, 1, 1)
        self.ori_conv = nn.Conv2d(channel, channel, 1)
        self.atr2_conv = conv3x3_bn_relu(channel, channel, padding=2, dilate=2)
        self.atr3_conv = conv3x3_bn_relu(channel, channel, padding=3, dilate=3)
        self.atr5_conv = conv3x3_bn_relu(channel, channel, padding=5, dilate=5)
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(channel, channel, 1)
        )
        self.fuse_atr3 = nn.Conv2d(channel * 2, channel, 1)
        self.fuse_atr5 = nn.Conv2d(channel * 3, channel, 1)
        self.fuse_conv = nn.Conv2d(channel * 5, channel, 1)

    def forward(self, x):
        I1 = self.ori_conv(x)
        I2 = self.atr2_conv(x)
        atr3 = self.atr3_conv(x)
        atr5 = self.atr5_conv(x)
        I3 = self.fuse_atr3(torch.cat([I2, atr3], 1))
        I4 = self.fuse_atr5(torch.cat([I2, I3, atr5], 1))
        I5 = self.pool_conv(x)
        return self.self_conv(x) + self.fuse_conv(torch.cat([I1, I2, I3, I4, I5], 1))


if __name__ == '__main__':
    input = torch.rand(2, 64, 32, 32)
    model = MFIM(64)
    out = model(input)
    print(out.shape)



import torch
from torch import nn


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m,
                        (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d,
                         nn.AdaptiveMaxPool2d, nn.SiLU, nn.LeakyReLU,
                         nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity, nn.GELU)):
            pass
        else:
            m.initialize()


def conv3x3_bn(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
        nn.BatchNorm2d(out_planes),
    )


def conv3x3_bn_relu(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def conv3x3_bn_pRelu(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(),
    )


def conv3x3_bn_gelu(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.GELU()
    )


def conv3x3_bn_silu(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.SiLU(inplace=True)
    )


# 定义深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积步骤，使用3x3卷积核，不改变卷积后的输出尺寸
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels)
        # 逐点卷积步骤，增加通道数
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.initialize()

    def forward(self, x):
        # 先进行深度卷积
        x = self.depthwise(x)
        # 然后进行逐点卷积，增加通道数
        x = self.pointwise(x)
        return x

    def initialize(self):
        weight_init(self)

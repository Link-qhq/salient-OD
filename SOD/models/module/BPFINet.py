import torch
from torch import nn
import torch.nn.functional as F
from models.module.common import conv3x3_bn


class ChannelCompress(nn.Module):
    """
        通道压缩模块，放在decoder后进行通道压缩
        实质上用通道注意力机制
    """
    def __init__(self, in_channel, out_channel):
        super(ChannelCompress, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(nn.Linear(out_channel, out_channel // 4, bias=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(out_channel // 4, out_channel, bias=False), nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(out_channel))

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv(x)
        y1 = self.pool(x)
        y1 = y1.reshape((n, -1))
        y1 = self.linear1(y1)
        y1 = self.linear2(y1)
        y1 = y1.reshape((n, self.out_channel, 1, 1))

        y1 = y1.expand_as(x).clone()
        y = x * y1
        return F.relu(y + self.conv2(y))


class U_shapeSelfRefinement(nn.Module):
    """
        U型自细化模块 USRM
        某个卷积层只能捕获某一特定规模的信息
        显著对象的尺寸往往多种多样
    """
    def __init__(self, channel):
        super(U_shapeSelfRefinement, self).__init__()
        self.channel = channel
        self.conv1 = conv3x3_bn(self.channel, self.channel, stride=2, padding=1, dilate=1)
        self.conv2 = conv3x3_bn(self.channel, self.channel, stride=2, padding=1, dilate=1)
        self.conv3 = conv3x3_bn(self.channel, self.channel, stride=1, padding=2, dilate=2)

        self.conv_rev1 = conv3x3_bn(self.channel, self.channel, stride=1, padding=1, dilate=1)
        self.conv_rev2 = conv3x3_bn(self.channel, self.channel, stride=1, padding=1, dilate=1)

        self.conv_sum = conv3x3_bn(self.channel, self.channel, stride=1, padding=1, dilate=1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)

        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev1(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev2(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


"""
UFIM意为U-shape feature integration module，U型特征融合模块。那么为什么要做特征融合呢？
    高级特征包含更多的语义信息，有利于定位显著区域
    低级特征包含更多的细节信息，有利于定位显著边界
    在向网络浅层传递的过程中，全局特征会逐渐被稀释
UFIM的作用就是将高级特征、低级特征、全局特征给聚合起来。而说到特征聚合，最基础的一个方法就是像UNet那样，做一个skip connection，
把不同的特征直接concat或者相加。然而这么做的缺陷在于： 三种特征的感受野差别很大，直接resize再concat或add可能并不合适
既然感受野差别很大不好直接加，那就可以想办法把三者的感受野调整至差不多，然后再加，因此UFIM便通过相应的卷积层来完成感受野调整的过程。
针对不同层级的特征，UFIM的深度也有所不同，这里以最后一个UFIM为例进行说明。其中的X，Y，Z分别对应着低级特征、高级特征、全局特征：
"""


class U_shapeFeatureIntegration(nn.Module):
    def __init__(self, channel):
        super(U_shapeFeatureIntegration, self).__init__()
        self.channel = channel
        self.conv1 = conv3x3_bn(self.channel, self.channel, stride=2, padding=1, dilate=1)
        self.conv2 = conv3x3_bn(self.channel, self.channel, stride=2, padding=1, dilate=1)
        self.conv3 = conv3x3_bn(self.channel, self.channel, stride=2, padding=1, dilate=1)
        self.conv4 = conv3x3_bn(self.channel, self.channel, stride=2, padding=1, dilate=1)
        self.conv5 = conv3x3_bn(self.channel, self.channel, stride=1, padding=2, dilate=2)

        self.conv_rev1 = conv3x3_bn(self.channel, self.channel, stride=1, padding=1, dilate=1)
        self.conv_rev2 = conv3x3_bn(self.channel, self.channel, stride=1, padding=1, dilate=1)
        self.conv_rev3 = conv3x3_bn(self.channel, self.channel, stride=1, padding=1, dilate=1)
        self.conv_rev4 = conv3x3_bn(self.channel, self.channel, stride=1, padding=1, dilate=1)

        self.conv_sum = conv3x3_bn(self.channel, self.channel, stride=1, padding=1, dilate=1)

    def forward(self, x, high, gi):
        """
        :param x: 低级特征 第一个编码器特征 [bs, C, H / 2, W / 2]
        :param high: 高级特征 [bs, C, H / 4, W / 4]
        :param gi: 全局特征 [bs, C, H / 16, W / 16]
        :return: [bs, C, H / 2, W / 2]
        """
        # gi means global information
        y1 = self.conv1(x)
        y1 = y1 + F.interpolate(high, y1.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv2(y1)
        y2 = y2 + F.interpolate(gi, y2.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)

        y4up = F.interpolate(y5, y4.shape[2:], mode='bilinear', align_corners=True)
        y4 = self.conv_rev1(y4 + y4up)
        y3up = F.interpolate(y4, y3.shape[2:], mode='bilinear', align_corners=True)
        y3 = self.conv_rev2(y3 + y3up)
        y2up = F.interpolate(y3, y2.shape[2:], mode='bilinear', align_corners=True)
        y2 = self.conv_rev3(y2 + y2up)
        y1up = F.interpolate(y2, y1.shape[2:], mode='bilinear', align_corners=True)
        y1 = self.conv_rev4(y1 + y1up)
        y = F.interpolate(y1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.conv_sum(F.relu(x + y))


if __name__ == '__main__':
    input1 = torch.rand(1, 64, 112, 112)
    input2 = torch.rand(1, 64, 64, 64)
    input3 = torch.rand(1, 64, 7, 7)
    net1 = U_shapeFeatureIntegration(64)
    print(net1(input1, input2, input3).shape)

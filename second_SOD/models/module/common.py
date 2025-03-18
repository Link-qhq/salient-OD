import torch
from torch import nn
import math


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


def conv3x3_bn_relu(in_planes, out_planes, stride=1, padding=1, dilate=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilate, bias=False, groups=groups),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def conv3x3_bn_pRelu(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(),
    )


def conv3x3_bn_LRelu(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, dilation=dilate, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(inplace=True)
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


def conv1x1_bn(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes)
    )


def conv1x1_bn_relu(in_planes, out_planes, stride=1, padding=1, dilate=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True)
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


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s


class ConvBNReLU(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
                 bias=True, use_relu=True, use_bn=True, frozen=False, residual=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=ksize, stride=stride, padding=pad, \
                              dilation=dilation, groups=groups, bias=bias)
        self.residual = residual
        if use_bn:
            if frozen:
                self.bn = FrozenBatchNorm2d(nOut)
            else:
                self.bn = nn.BatchNorm2d(nOut)
        else:
            self.bn = None
        if use_relu:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x1 = self.conv(x)
        if self.bn is not None:
            x1 = self.bn(x1)
        if self.residual and x1.shape[1] == x.shape[1]:
            x1 = x + x1
        if self.act is not None:
            x1 = self.act(x1)

        return x1


class ResidualConvBlock(nn.Module):
    def __init__(self, nIn, nOut, ksize=3, stride=1, pad=1, dilation=1, groups=1,
                 bias=True, use_relu=True, use_bn=True, frozen=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = ConvBNReLU(nIn, nOut, ksize=ksize, stride=stride, pad=pad,
                               dilation=dilation, groups=groups, bias=bias,
                               use_relu=use_relu, use_bn=use_bn, frozen=frozen)
        self.residual_conv = ConvBNReLU(nIn, nOut, ksize=1, stride=stride, pad=0,
                                        dilation=1, groups=groups, bias=bias,
                                        use_relu=False, use_bn=use_bn, frozen=frozen)

    def forward(self, x):
        x = self.conv(x) + self.residual_conv(x)
        return x


class ReceptiveConv(nn.Module):
    def __init__(self, inplanes, planes, baseWidth=24, scale=4, dilation=None, aggregation=True, use_dwconv=False):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(ReceptiveConv, self).__init__()
        assert scale >= 1, 'The input scale must be a positive value'

        self.width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, self.width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.width * scale)
        # self.nums = 1 if scale == 1 else scale - 1
        self.nums = scale

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        dilation = [1] * self.nums if dilation is None else dilation
        for i in range(self.nums):
            if use_dwconv:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3, padding=dilation[i], dilation=dilation[i], groups=self.width, bias=False))
            else:
                self.convs.append(nn.Conv2d(self.width, self.width, kernel_size=3,
                                            padding=dilation[i], dilation=dilation[i], bias=False))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.conv3 = nn.Conv2d(self.width * scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.scale = scale
        self.aggregation = aggregation

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if self.aggregation:
                sp = spx[i] if i == 0 else sp + spx[i]
            else:
                sp = spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x
        out = self.relu(out)

        return out

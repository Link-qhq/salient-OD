import torch
from torch import nn
from models.module.common import conv3x3_bn_pRelu
from torch.nn import functional as F


class DPConv(nn.Module):
    def __init__(self, channel):
        super(DPConv, self).__init__()
        self.channel = channel
        self.conv = nn.Sequential(nn.Conv2d(channel, channel, 1, 1), nn.BatchNorm2d(channel), nn.PReLU())
        self.alpha = nn.Sequential(
            nn.Conv2d(channel, channel * 2, 1, 1), nn.ReLU(True),
            nn.Conv2d(channel * 2, channel * 4, 1, 1), nn.Softmax(1)
        )
        self.conv3 = conv3x3_bn_pRelu(channel, channel)
        self.conv5 = nn.Sequential(nn.Conv2d(channel, channel, 5, 1, 2, groups=channel), nn.BatchNorm2d(channel), nn.PReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(channel, channel, 7, 1, 3, groups=channel), nn.BatchNorm2d(channel), nn.PReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(channel, channel, 9, 1, 4, groups=channel), nn.BatchNorm2d(channel), nn.PReLU())
        self.conv_out = nn.Sequential(nn.Conv2d(channel * 4, channel, 1, 1), nn.BatchNorm2d(channel), nn.PReLU())

    def forward(self, x):
        y = self.conv(x)
        gap = F.avg_pool2d(y, kernel_size=(y.size(2), y.size(3)))
        x3 = self.conv3(y)
        x5 = self.conv5(y)
        x7 = self.conv7(y)
        x9 = self.conv9(y)
        alpha1, alpha2, alpha3, alpha4 = torch.split(self.alpha(gap), self.channel, 1)
        x3 = x3 * alpha1
        x5 = x5 * alpha2
        x7 = x7 * alpha3
        x9 = x9 * alpha4
        return x + self.conv_out(torch.cat([x3, x5, x7, x9], 1))


class DWF(nn.Module):
    def __init__(self, channel, level=1):
        super(DWF, self).__init__()
        self.channel = channel
        self.level = level
        self.theta = nn.Sequential(
            nn.Conv2d(channel * 4, channel, 1, 1), nn.ReLU(True),
            nn.Conv2d(channel, channel * 4, 1, 1), nn.Softmax(1)
        )

    def forward(self, fea):
        f1 = F.interpolate(fea[0], size=fea[self.level].shape[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(fea[1], size=fea[self.level].shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(fea[2], size=fea[self.level].shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(fea[3], size=fea[self.level].shape[2:], mode='bilinear', align_corners=True)
        gap = F.max_pool2d(torch.cat([f1, f2, f3, f4], 1), kernel_size=(f1.size(2), f1.size(3)), stride=(f1.size(2), f1.size(3)))
        theta1, theta2, theta3, theta4 = torch.split(self.theta(gap), self.channel, 1)
        return f1 * theta1 + f2 * theta2 + f3 * theta3 + f4 * theta4

if __name__ == '__main__':
    x = torch.rand(1, 64, 32, 32)
    model = DPConv(64)
    out = model(x)
    print(out.shape)

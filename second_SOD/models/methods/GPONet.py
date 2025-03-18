import torch
from torch import nn
from torch.nn import functional as F
from models.module.common import conv3x3_bn_relu


# Gate Fusion Unit
# 以互补的方式融合两个相邻级别的特征，以避免冗余信息传输
class GFU(nn.Module):
    def __init__(self, channel):
        super(GFU, self).__init__()
        self.gate1 = conv3x3_bn_relu(channel, channel)  # for lower feat        [B, channel, M, N]
        self.gate2 = conv3x3_bn_relu(channel, channel)  # for higher feat       [B, channel, M, N]
        self.convAct1 = conv3x3_bn_relu(channel, channel)  # for higher feat       [B, channel, M, N]
        self.convAct2 = conv3x3_bn_relu(channel, channel)  # for higher feat       [B, channel, M, N]
        self.convLack = conv3x3_bn_relu(channel, channel)  # for higher feat       [B, channel, M, N]
        self.fuse_conv = conv3x3_bn_relu(channel, channel)

    def forward(self, high, low):  # H//2 H <=> 高级特征 低级特征
        high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=True)
        g2 = self.gate2(high)
        act2 = self.convAct2(g2 * high)

        g1 = self.gate1(low)
        act1 = self.convAct1(g1 * low)
        hsl = self.convLack((1. - g1) * act2)
        out = hsl + act1 + hsl
        return self.fuse_conv(out)


class GFN(nn.Module):
    def __init__(self, channel_list=None, channel=64):
        super(GFN, self).__init__()
        if channel_list is None:
            channel_list = [256, 512, 1024, 2048]
        self.down1 = conv3x3_bn_relu(channel_list[0], channel)
        self.down2 = conv3x3_bn_relu(channel_list[1], channel)
        self.down3 = conv3x3_bn_relu(channel_list[2], channel)
        self.down4 = conv3x3_bn_relu(channel_list[3], channel)

        self.gfu1 = GFU(channel)
        self.gfu2 = GFU(channel)
        self.gfu3 = GFU(channel)

    def forward(self, fea):
        fea1 = self.down1(fea[0])
        fea2 = self.down2(fea[1])
        fea3 = self.down3(fea[2])
        fea4 = self.down4(fea[3])
        fea3 = self.gfu3(fea4, fea3)
        fea2 = self.gfu2(fea3, fea2)
        fea1 = self.gfu1(fea2, fea1)
        return fea1, fea2, fea3, fea4


# Cross guide module
# 引导两个分支之间的相互交流，充分利用各层次上全局特征和细节特征的复杂语义信息
class CGM(nn.Module):
    def __init__(self, channel):
        super(CGM, self).__init__()
        # dt
        self.conv1_1 = conv3x3_bn_relu(channel, channel)
        self.conv1_2 = conv3x3_bn_relu(channel, channel)
        # gb
        self.conv2_1 = conv3x3_bn_relu(channel, channel)
        self.conv2_2 = conv3x3_bn_relu(channel, channel)

        self.conv3_1 = conv3x3_bn_relu(channel * 2, channel)
        self.conv3_2 = conv3x3_bn_relu(channel * 2, channel)

        self.conv4_1 = conv3x3_bn_relu(channel, channel)
        self.conv4_2 = conv3x3_bn_relu(channel, channel)

    def forward(self, gb, dt):  # global and detail branches
        dt1, gb1 = self.conv1_1(dt), self.conv2_1(gb)
        dt2, gb2 = self.conv1_2(dt1), self.conv2_2(gb1)
        cro_feat = torch.cat((dt2, gb2), dim=1)
        dt3, gb3 = self.conv3_1(cro_feat), self.conv3_2(cro_feat)
        f1 = dt1 + dt3
        f2 = gb1 + gb3
        dt4 = self.conv4_1(f1)
        gb4 = self.conv4_2(f2)
        return gb4, dt4


# Feature fusion module
# 考虑由全局和细节分支输出的多个特征图的不同贡献
class FFM(nn.Module):
    def __init__(self, fuse_channel=256, num_pred=8):
        super(FFM, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Sequential(nn.Linear(num_pred, fuse_channel),
                                  nn.BatchNorm1d(fuse_channel),
                                  nn.ReLU(),
                                  nn.Linear(fuse_channel, num_pred))

    def forward(self, input):
        B = input.shape[0]

        # get fuse attention
        gap = self.GAP(input).squeeze(-1).squeeze(-1)
        fuse_att = self.fuse(gap).view(B, 8, 1, 1)

        # fuse from gb&dt out
        fuse = input * fuse_att.expand_as(input)

        return fuse


if __name__ == '__main__':
    input1 = torch.rand(2, 256, 56, 56)
    input2 = torch.rand(2, 512, 28, 28)
    input3 = torch.rand(2, 1024, 14, 14)
    input4 = torch.rand(2, 2048, 7, 7)
    model = GFN()
    out = model((input1, input2, input3, input4))
    # print(out.shape)
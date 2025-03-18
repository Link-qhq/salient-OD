import torch
from torch import nn
from models.module.resnet import ResNet
from models.module.ICONet import asyConv
import torch.nn.functional as F
from models.module.attention import CBAM, SpatialAttention, SpaceAttention
from models.module.CSEPNet import CSFI
from models.module.common import weight_init
from models.module.DSRNet import CCM
from models.module.SPP import SPPF_LSKA
from models.module.FP import iAFF


# DBB--net
class DFA(nn.Module):
    """
        Enhance the feature diversity.
        多样性特征融合模块
    """

    def __init__(self, x, y, dilation=2):
        """
        :param x: 输入通道数
        :param y: 输出通道数
        """
        super(DFA, self).__init__()
        self.asyConv = nn.Sequential(
            asyConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                    padding_mode='zeros', deploy=False)
        )
        self.oriConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(y)
        )
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=dilation, padding=dilation, stride=1),
            nn.BatchNorm2d(y)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(y)
        )
        self.conv1_k = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(y),
            nn.Conv2d(y, y, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(y)
        )
        self.conv1_p = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(y),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(y)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(y, y, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(y),
            nn.LeakyReLU()
        )

        self.initialize()

    def forward(self, f):
        p1 = self.oriConv(f)  # 原始卷积
        p2 = self.asyConv(f)  # 非对称卷积
        p3 = self.atrConv(f)  # 空洞卷积
        p4 = self.conv1_p(f)  # 1x1 + avgPool
        p5 = self.conv1_k(f)  # 1x1 + 3x3
        p6 = self.conv1x1(f)  # 1x1
        p = p1 + p2 + p3 + p4 + p5 + p6
        p = self.conv(p)
        return p

    def initialize(self):
        # pass
        weight_init(self)


class BIG(nn.Module):
    """
        boundary information guidance 边界信息引导模块
    """

    def __init__(self, channel, groups=16):
        super(BIG, self).__init__()
        self.gate1 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate4 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                  nn.GroupNorm(groups, channel), nn.PReLU())

        self.channel = channel
        self.weight = nn.Softmax(dim=1)

    def forward(self, x, edge):
        """
        :param x: torch.Size([1, channel(↑), 56, 56])
        :param edge: torch.Size([1, 1, 56, 56])
        :return:
        """
        x1, x2, x3, x4 = torch.split(x, self.channel // 4, dim=1)

        cm1 = self.gate1(x1)
        cm2 = self.gate2(x2)
        cm3 = self.gate3(x3)
        cm4 = self.gate4(x4)

        e1 = cm1 * torch.sigmoid(edge)
        e2 = cm2 * torch.sigmoid(edge)
        e3 = cm3 * torch.sigmoid(edge)
        e4 = cm4 * torch.sigmoid(edge)

        gv1 = F.avg_pool2d(e1, (e1.size(2), e1.size(3)), stride=(e1.size(2), e1.size(3)))
        gv2 = F.avg_pool2d(e2, (e2.size(2), e2.size(3)), stride=(e2.size(2), e2.size(3)))
        gv3 = F.avg_pool2d(e3, (e3.size(2), e3.size(3)), stride=(e3.size(2), e3.size(3)))
        gv4 = F.avg_pool2d(e4, (e4.size(2), e4.size(3)), stride=(e4.size(2), e4.size(3)))

        weight = self.weight(torch.cat((gv1, gv2, gv3, gv4), 1))
        w1, w2, w3, w4 = torch.split(weight, 1, dim=1)

        nx1 = x1 * w1
        nx2 = x2 * w2
        nx3 = x3 * w3
        nx4 = x4 * w4

        return self.conv(torch.cat((nx1, nx2, nx3, nx4), 1))


class SDU(nn.Module):
    def __init__(self, channel=256, groups=32):
        super(SDU, self).__init__()
        if channel % groups != 0:
            assert 'groups must be divisible channel'
        # SDU
        self.pr1_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(groups, channel), nn.PReLU()
        )
        self.pr1_2 = nn.Sequential(
            CCM(channel, channel // 2, redio=4),
            nn.Conv2d(channel // 2, 1, 1)
        )

        self.pe1_1 = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=3, padding=1), nn.GroupNorm(groups, channel), nn.PReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(groups, channel), nn.PReLU()
        )
        self.pe1_2 = nn.Sequential(
            CCM(channel, channel // 2, 4),
            nn.Conv2d(channel // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.pr1_1(x)
        out2 = self.pr1_2(out1)
        out3 = self.pe1_1(out2)
        out4 = self.pe1_2(out3)
        return out1, out2, out3, out4


class ICE(nn.Module):
    def __init__(self, num_channels=64, ratio=8):
        """
            空间通道完整性增强模块
        :param num_channels:
        :param ratio:
        """
        super(ICE, self).__init__()
        self.conv_cross = nn.Conv2d(3 * num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_cross = nn.BatchNorm2d(num_channels)

        self.eps = 1e-5

        self.conv_mask = nn.Conv2d(num_channels, 1, kernel_size=1)  # context Modeling
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // ratio, kernel_size=1),
            nn.LayerNorm([num_channels // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // ratio, num_channels, kernel_size=1)
        )
        self.space_add_conv = SpaceAttention()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.initialize()

    def forward(self, in1, in2=None, in3=None):
        if in2 is not None and in1.size()[2:] != in2.size()[2:]:
            in2 = F.interpolate(in2, size=in1.size()[2:], mode='bilinear')
        else:
            in2 = in1
        if in3 is not None and in1.size()[2:] != in3.size()[2:]:
            in3 = F.interpolate(in3, size=in1.size()[2:], mode='bilinear')
        else:
            in3 = in1

        x = torch.cat((in1, in2, in3), 1)
        x = F.relu(self.bn_cross(self.conv_cross(x)))  # [B, C, H, W]

        context1 = (x.pow(2).sum((2, 3), keepdim=True) + self.eps).pow(0.5)  # [B, C, 1, 1]  L2正则化
        channel_add_term = self.channel_add_conv(context1)
        out1 = x * channel_add_term

        context2 = (x.pow(2).sum(1, keepdims=True) + self.eps).pow(0.5)  # [B, 1, H, W] L2 正则化
        space_add_term = self.space_add_conv(context2)
        out2 = x * space_add_term

        out = self.conv(torch.cat([out1, out2], 1))
        return out

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        set_channels = 64
        self.dfa0 = DFA(64, set_channels)
        self.dfa1 = DFA(256, set_channels)
        self.dfa2 = DFA(512, set_channels)
        self.dfa3 = DFA(1024, set_channels)
        self.dfa4 = DFA(2048, set_channels)

        self.sppf = SPPF_LSKA(2048, set_channels, 5)
        self.cbam1 = CBAM(256, 16, 7)  # 融合改进后的dfa
        self.cbam2 = CBAM(512, 16, 7)
        self.cbam3 = CBAM(1024, 16, 7)
        self.cbam4 = CBAM(2048, 16, 7)

        self.iaff = iAFF(set_channels)

        self.ice1 = ICE(set_channels)
        self.ice2 = ICE(set_channels)
        self.ice3 = ICE(set_channels)
        self.ice4 = ICE(set_channels)

        self.big0 = BIG(set_channels)
        self.big1 = BIG(set_channels)
        self.big2 = BIG(set_channels)
        self.big3 = BIG(set_channels)

        self.csfi1 = CSFI(set_channels, set_channels // 2)
        self.csfi2 = CSFI(set_channels, set_channels // 2)
        self.csfi3 = CSFI(set_channels, set_channels // 2)
        self.sdu0 = SDU(set_channels, set_channels // 4)
        self.sdu1 = SDU(set_channels, set_channels // 4)
        self.sdu2 = SDU(set_channels, set_channels // 4)
        self.sdu3 = SDU(set_channels, set_channels // 4)
        self.sdu4 = SDU(set_channels, set_channels // 4)

    def forward(self, fea, shape):
        fea_1, fea_2, fea_3, fea_4 = fea
        sppf = self.sppf(fea_4)
        fea_1 = self.dfa1(self.cbam1(fea_1))
        fea_2 = self.dfa2(self.cbam2(fea_2))
        fea_3 = self.dfa3(self.cbam3(fea_3))
        fea_4 = self.dfa4(self.cbam4(fea_4))

        fea_4 = self.iaff(fea_4, sppf)
        out4_1, out4_2, out4_3, out4_4 = self.sdu4(fea_4)

        out3 = self.ice4(fea_3, out4_1, out4_3)
        out3 = self.big3(out3, F.interpolate(out4_4, size=out3.shape[2:], mode="bilinear"))
        out3 = self.csfi3(out3)

        out3_1, out3_2, out3_3, out3_4 = self.sdu3(out3)

        out2 = self.ice3(fea_2, out3_1, out3_3)
        out2 = self.big2(out2, F.interpolate(out3_4, size=out2.shape[2:], mode="bilinear"))
        out2 = self.csfi2(out2)

        out2_1, out2_2, out2_3, out2_4 = self.sdu2(out2)

        out1 = self.ice2(fea_1, out2_1, out2)
        out1 = self.big1(out1, F.interpolate(out2_4, size=out1.shape[2:], mode="bilinear"))
        out1 = self.csfi1(out1)

        out1_1, out1_2, out1_3, out1_4 = self.sdu1(out1)

        out1 = F.interpolate(out1_2, size=shape, mode="bilinear", align_corners=True)
        out2 = F.interpolate(out2_2, size=shape, mode="bilinear", align_corners=True)
        out3 = F.interpolate(out3_2, size=shape, mode="bilinear", align_corners=True)
        out4 = F.interpolate(out4_2, size=shape, mode="bilinear", align_corners=True)

        out1_1 = F.interpolate(out1_4, size=shape, mode="bilinear", align_corners=True)
        out2_1 = F.interpolate(out2_4, size=shape, mode="bilinear", align_corners=True)
        out3_1 = F.interpolate(out3_4, size=shape, mode="bilinear", align_corners=True)
        out4_1 = F.interpolate(out4_4, size=shape, mode="bilinear", align_corners=True)

        return out4, out3, out2, out1, out4_1, out3_1, out2_1, out1_1


class TestMODEL(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_size=224):
        super(TestMODEL, self).__init__()
        self.img_size = [img_size, img_size]
        self.encoder = ResNet()
        self.decoder = Decoder()

    def forward(self, x, shape=None):
        fea = self.encoder(x)
        out = self.decoder(fea, x.shape[2:] if shape is None else shape)
        return out


if __name__ == '__main__':
    from thop import profile

    input = torch.rand(2, 3, 224, 224)
    model = TestMODEL(3, 1)
    ouput = model(input)
    for i in range(8):
        print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    # a = torch.rand(1, 256, 56, 56)
    # net = DFA(256, 64)
    # print(net(a).shape)

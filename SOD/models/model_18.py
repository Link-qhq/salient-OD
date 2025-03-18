import torch
from torch import nn
from models.module.resnet import ResNet
from models.module.ICONet import asyConv
import torch.nn.functional as F
from models.module.attention import CBAM, SpaceAttention
from models.module.CSEPNet import CSFI
from models.module.common import weight_init
from models.module.DSRNet import CCM
from models.module.SPP import SPPF_LSKA
from models.module.FP import iAFF


# 消融实验
# 单通道分支
# 单空间分支
# 级联分支
class MSFE(nn.Module):
    """
        Multi-scale feature extract
    """
    def __init__(self, in_ch, out_ch):
        super(MSFE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=5, padding=5)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        global_avgPool = torch.sigmoid(F.relu(F.avg_pool2d(x, (H, W))))
        x += (x * global_avgPool)
        return x


class DFFM(nn.Module):
    """
        Diversity Feature Fusion Module
    """

    def __init__(self, in_ch, out_ch, dilation=2):
        super(DFFM, self).__init__()
        self.att = CBAM(in_ch, reduction=16, kernel_size=7)
        self.asyConv = nn.Sequential(
            asyConv(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                    padding_mode='zeros', deploy=False)
        )
        self.oriConv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.atrConv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=dilation, padding=dilation, stride=1),
            nn.BatchNorm2d(out_ch)
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch)
        )
        self.conv1_k = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.conv1_p = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )
        self.initialize()

    def forward(self, f):
        f = self.att(f)  # 注意力提纯
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
        weight_init(self)


class SCEM(nn.Module):
    """
        Space-Channel Enhanced Module
    """

    def __init__(self, num_channels=64, ratio=8):
        super(SCEM, self).__init__()
        # self.conv_cross = nn.Conv2d(3 * num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn_cross = nn.BatchNorm2d(num_channels)
        self.conv_cross = nn.Sequential(
            nn.Conv2d(3 * num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.eps = 1e-5

        self.conv_mask = nn.Conv2d(num_channels, 1, kernel_size=1)  # context Modeling
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // ratio, kernel_size=1),
            nn.LayerNorm([num_channels // ratio, 1, 1]),
            nn.LeakyReLU(inplace=True),
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
        # x = F.relu(self.bn_cross(self.conv_cross(x)))  # [B, C, H, W]
        x = self.conv_cross(x)
        # 级联消融实验
        # 通道注意力
        context1 = (x.pow(2).sum((2, 3), keepdim=True) + self.eps).pow(0.5)  # [B, C, 1, 1]  L2正则化
        channel_add_term = self.channel_add_conv(context1)
        x *= channel_add_term
        # 空间注意力
        context2 = (x.pow(2).sum(1, keepdims=True) + self.eps).pow(0.5)  # [B, 1, H, W] L2 正则化
        space_add_term = self.space_add_conv(context2)
        out2 = x * space_add_term

        # out = self.conv(out2)

        return out2

    def initialize(self):
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
        self.fusion1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
        self.fusion4 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1)
        )
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

        c1 = self.gate1(x1)
        c2 = self.gate2(x2)
        c3 = self.gate3(x3)
        c4 = self.gate4(x4)
        e1 = c1 * torch.sigmoid(edge)
        e2 = c2 * torch.sigmoid(edge)
        e3 = c3 * torch.sigmoid(edge)
        e4 = c4 * torch.sigmoid(edge)

        gv1 = F.avg_pool2d(e1, (e1.size(2), e1.size(3)), stride=(e1.size(2), e1.size(3)))  # [B, 1, 1, 1]
        gv2 = F.avg_pool2d(e2, (e2.size(2), e2.size(3)), stride=(e2.size(2), e2.size(3)))
        gv3 = F.avg_pool2d(e3, (e3.size(2), e3.size(3)), stride=(e3.size(2), e3.size(3)))
        gv4 = F.avg_pool2d(e4, (e4.size(2), e4.size(3)), stride=(e4.size(2), e4.size(3)))

        gm1 = F.max_pool2d(e1, (e1.size(2), e1.size(3)), stride=(e1.size(2), e1.size(3)))
        gm2 = F.max_pool2d(e2, (e2.size(2), e2.size(3)), stride=(e2.size(2), e2.size(3)))
        gm3 = F.max_pool2d(e3, (e3.size(2), e3.size(3)), stride=(e3.size(2), e3.size(3)))
        gm4 = F.max_pool2d(e4, (e4.size(2), e4.size(3)), stride=(e4.size(2), e4.size(3)))

        g1 = self.fusion1(torch.cat([gv1, gm1], dim=1))
        g2 = self.fusion2(torch.cat([gv2, gm2], dim=1))
        g3 = self.fusion3(torch.cat([gv3, gm3], dim=1))
        g4 = self.fusion4(torch.cat([gv4, gm4], dim=1))

        weight = self.weight(torch.cat([g1, g2, g3, g4], dim=1))
        w1, w2, w3, w4 = torch.split(weight, 1, 1)
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


class FSDM(nn.Module):
    """
        Fine Salient-map Purification module
    """

    def __init__(self, num_channels=64, group=16):
        super(FSDM, self).__init__()
        self.big = BIG(num_channels, group)
        self.cfsi = CSFI(num_channels, num_channels // 2)
        self.sdu = SDU(num_channels, group)

    def forward(self, x, edge=None):
        if edge is not None:
            x = self.big(x, F.interpolate(edge, size=x.shape[2:], mode="bilinear"))
            x = self.cfsi(x)
        out1, out2, out3, out4 = self.sdu(x)
        return out1, out2, out3, out4


class MFSDM(nn.Module):
    """
        modify FSDM
    """
    def __init__(self, in_ch, ratio=4, group=16):
        super(MFSDM, self).__init__()
        inter_channels = in_ch // ratio
        # 局部注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(in_ch, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, in_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_ch)
        )
        # 全局注意力
        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, in_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_ch)
        )
        # 局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(in_ch, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, in_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_ch)
        )
        # 全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, in_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_ch)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, in_ch, kernel_size=1)
        )
        self.simoid = nn.Sigmoid()

        self.sdu1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group, in_ch),
            nn.PReLU()
        )
        self.sdu2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group, in_ch),
            nn.PReLU()
        )
        self.head1 = nn.Sequential(
            CCM(in_ch, in_ch // 2, redio=4),
            nn.Conv2d(in_ch // 2, 1, 1)
        )
        self.head2 = nn.Sequential(
            CCM(in_ch, in_ch // 2, redio=4),
            nn.Conv2d(in_ch // 2, 1, 1)
        )

    def forward(self, x, edge=None):
        if edge is not None:
            edge = F.interpolate(edge, size=x.shape[2:], mode="bilinear")
            x1 = x - edge  # 前背景
            x2 = x + edge  # 边界

            x1_1 = self.local_att1(x1)
            x1_2 = self.global_att1(x1)
            wei1 = self.simoid(x1_1 + x1_2)

            x2_1 = self.local_att2(x2)
            x2_2 = self.global_att2(x2)
            wei2 = self.simoid(x2_1 + x2_2)

            x = self.conv(x1 * wei1 + x2 * wei2)

        out1_1 = self.sdu1(x)
        out1_2 = self.head1(out1_1)

        out2_1 = self.sdu2(x)
        out2_2 = self.head2(out2_1)

        return out1_1, out1_2, out2_1, out2_2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        set_channels = 64
        ratio = 8
        group = 16

        self.dff1 = DFFM(256, set_channels)
        self.dff2 = DFFM(512, set_channels)
        self.dff3 = DFFM(1024, set_channels)
        self.dff4 = DFFM(2048, set_channels)

        self.spp = SPPF_LSKA(2048, set_channels, 5)
        self.iffa = iAFF(set_channels)
        self.sce1 = SCEM(set_channels, ratio)
        self.sce2 = SCEM(set_channels, ratio)
        self.sce3 = SCEM(set_channels, ratio)
        self.sce4 = SCEM(set_channels, ratio)

        self.fsd1 = MFSDM(set_channels, ratio, group)
        self.fsd2 = MFSDM(set_channels, ratio, group)
        self.fsd3 = MFSDM(set_channels, ratio, group)
        self.fsd4 = MFSDM(set_channels, ratio, group)

    def forward(self, fea, shape):
        fea1, fea2, fea3, fea4 = fea

        spp = self.spp(fea4)

        fea1 = self.dff1(fea1)
        fea2 = self.dff2(fea2)
        fea3 = self.dff3(fea3)
        fea4 = self.dff4(fea4)

        fea4 = self.iffa(fea4, spp)
        out4_1, out4_2, out4_3, out4_4 = self.fsd4(fea4)
        fea3 = self.sce3(fea3, out4_1, out4_3)
        out3_1, out3_2, out3_3, out3_4 = self.fsd3(fea3, out4_4)
        fea2 = self.sce2(fea2, out3_1, out3_3)
        out2_1, out2_2, out2_3, out2_4 = self.fsd2(fea2, out3_4)
        fea1 = self.sce1(fea1, out2_1, out2_3)
        out1_1, out1_2, out1_3, out1_4 = self.fsd1(fea1, out2_4)

        out1 = F.interpolate(out1_2, size=shape, mode="bilinear", align_corners=True)
        out2 = F.interpolate(out2_2, size=shape, mode="bilinear", align_corners=True)
        out3 = F.interpolate(out3_2, size=shape, mode="bilinear", align_corners=True)
        out4 = F.interpolate(out4_2, size=shape, mode="bilinear", align_corners=True)

        out1_1 = F.interpolate(out1_4, size=shape, mode="bilinear", align_corners=True)
        out2_1 = F.interpolate(out2_4, size=shape, mode="bilinear", align_corners=True)
        out3_1 = F.interpolate(out3_4, size=shape, mode="bilinear", align_corners=True)
        out4_1 = F.interpolate(out4_4, size=shape, mode="bilinear", align_corners=True)

        return out4, out3, out2, out1, out4_1, out3_1, out2_1, out1_1


class PGN(nn.Module):
    """
        Detail-guided salient object detection network
        Progressively guided network of detailed information
    """

    def __init__(self, in_channels=3, out_channels=1, img_size=224):
        super(PGN, self).__init__()
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
    edge = torch.rand(2, 1, 56, 56)
    model = PGN()
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
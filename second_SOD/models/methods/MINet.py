import torch
from torch import nn
from torch.nn import functional as F

from models.backbone.resnet_2 import resnet50


class SIM(nn.Module):
    def __init__(self, h_C, l_C):
        super(SIM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = cus_sample

        self.h2l_0 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.h2h_0 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.bnl_0 = nn.BatchNorm2d(l_C)
        self.bnh_0 = nn.BatchNorm2d(h_C)

        self.h2h_1 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(h_C, l_C, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(l_C)
        self.bnh_1 = nn.BatchNorm2d(h_C)

        self.h2h_2 = nn.Conv2d(h_C, h_C, 3, 1, 1)
        self.l2h_2 = nn.Conv2d(l_C, h_C, 3, 1, 1)
        self.bnh_2 = nn.BatchNorm2d(h_C)

        self.relu = nn.ReLU(True)

    def forward(self, x):
        h, w = x.shape[2:]

        # first conv
        x_h = self.relu(self.bnh_0(self.h2h_0(x)))
        x_l = self.relu(self.bnl_0(self.h2l_0(self.h2l_pool(x))))

        # mid conv
        x_h2h = self.h2h_1(x_h)
        x_h2l = self.h2l_1(self.h2l_pool(x_h))
        x_l2l = self.l2l_1(x_l)
        x_l2h = self.l2h_1(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(self.l2h_up(x_l, size=(h, w)))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h + x


class conv_2nV1(nn.Module):
    def __init__(self, in_hc=64, in_lc=256, out_c=64, main=0):
        super(conv_2nV1, self).__init__()
        self.main = main
        mid_c = min(in_hc, in_lc)
        self.relu = nn.ReLU(True)
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2, mode="nearest")

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        self.bnh_1 = nn.BatchNorm2d(mid_c)

        if self.main == 0:
            # stage 2
            self.h2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2h_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnh_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.h2h_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnh_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_hc, out_c, 1)

        elif self.main == 1:
            # stage 2
            self.h2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.l2l_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
            self.bnl_2 = nn.BatchNorm2d(mid_c)

            # stage 3
            self.l2l_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
            self.bnl_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_lc, out_c, 1)

        else:
            raise NotImplementedError

    def forward(self, in_h, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        h2l = self.h2l_1(self.h2l_pool(h))
        l2l = self.l2l_1(l)
        l2h = self.l2h_1(self.l2h_up(l))
        h = self.relu(self.bnh_1(h2h + l2h))
        l = self.relu(self.bnl_1(l2l + h2l))

        if self.main == 0:
            # stage 2
            h2h = self.h2h_2(h)
            l2h = self.l2h_2(self.l2h_up(l))
            h_fuse = self.relu(self.bnh_2(h2h + l2h))

            # stage 3
            out = self.relu(self.bnh_3(self.h2h_3(h_fuse)) + self.identity(in_h))
            # 这里使用的不是in_h，而是h
        elif self.main == 1:
            # stage 2
            h2l = self.h2l_2(self.h2l_pool(h))
            l2l = self.l2l_2(l)
            l_fuse = self.relu(self.bnl_2(h2l + l2l))

            # stage 3
            out = self.relu(self.bnl_3(self.l2l_3(l_fuse)) + self.identity(in_l))
        else:
            raise NotImplementedError

        return out


class conv_3nV1(nn.Module):
    def __init__(self, in_hc=64, in_mc=256, in_lc=512, out_c=64):
        super(conv_3nV1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.downsample = nn.AvgPool2d((2, 2), stride=2)

        mid_c = min(in_hc, in_mc, in_lc)
        self.relu = nn.ReLU(True)

        # stage 0
        self.h2h_0 = nn.Conv2d(in_hc, mid_c, 3, 1, 1)
        self.m2m_0 = nn.Conv2d(in_mc, mid_c, 3, 1, 1)
        self.l2l_0 = nn.Conv2d(in_lc, mid_c, 3, 1, 1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        # stage 1
        self.h2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.h2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2h_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2l_1 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnh_1 = nn.BatchNorm2d(mid_c)
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        self.bnl_1 = nn.BatchNorm2d(mid_c)

        # stage 2
        self.h2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.l2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.m2m_2 = nn.Conv2d(mid_c, mid_c, 3, 1, 1)
        self.bnm_2 = nn.BatchNorm2d(mid_c)

        # stage 3
        self.m2m_3 = nn.Conv2d(mid_c, out_c, 3, 1, 1)
        self.bnm_3 = nn.BatchNorm2d(out_c)

        self.identity = nn.Conv2d(in_mc, out_c, 1)

    def forward(self, in_h, in_m, in_l):
        # stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m = self.relu(self.bnm_0(self.m2m_0(in_m)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        # stage 1
        h2h = self.h2h_1(h)
        m2h = self.m2h_1(self.upsample(m))

        h2m = self.h2m_1(self.downsample(h))
        m2m = self.m2m_1(m)
        l2m = self.l2m_1(self.upsample(l))

        m2l = self.m2l_1(self.downsample(m))
        l2l = self.l2l_1(l)

        h = self.relu(self.bnh_1(h2h + m2h))
        m = self.relu(self.bnm_1(h2m + m2m + l2m))
        l = self.relu(self.bnl_1(m2l + l2l))

        # stage 2
        h2m = self.h2m_2(self.downsample(h))
        m2m = self.m2m_2(m)
        l2m = self.l2m_2(self.upsample(l))
        m = self.relu(self.bnm_2(h2m + m2m + l2m))

        # stage 3
        out = self.relu(self.bnm_3(self.m2m_3(m)) + self.identity(in_m))
        return out


class AIM(nn.Module):
    def __init__(self, iC_list, oC_list):
        super(AIM, self).__init__()
        ic0, ic1, ic2, ic3, ic4 = iC_list
        oc0, oc1, oc2, oc3, oc4 = oC_list
        self.conv0 = conv_2nV1(in_hc=ic0, in_lc=ic1, out_c=oc0, main=0)
        self.conv1 = conv_3nV1(in_hc=ic0, in_mc=ic1, in_lc=ic2, out_c=oc1)
        self.conv2 = conv_3nV1(in_hc=ic1, in_mc=ic2, in_lc=ic3, out_c=oc2)
        self.conv3 = conv_3nV1(in_hc=ic2, in_mc=ic3, in_lc=ic4, out_c=oc3)
        self.conv4 = conv_2nV1(in_hc=ic3, in_lc=ic4, out_c=oc4, main=1)

    def forward(self, *xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        out_xs = []
        out_xs.append(self.conv0(xs[0], xs[1]))
        out_xs.append(self.conv1(xs[0], xs[1], xs[2]))
        out_xs.append(self.conv2(xs[1], xs[2], xs[3]))
        out_xs.append(self.conv3(xs[2], xs[3], xs[4]))
        out_xs.append(self.conv4(xs[3], xs[4]))

        return out_xs


def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y


def upsample_cat(*xs):
    y = xs[-1]
    out = []
    for x in xs[:-1]:
        out.append(F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False))
    return torch.cat([*out, y], dim=1)


def upsample_reduce(b, a):
    """
    上采样所有特征到最后一个特征的尺度以及前一个特征的通道数
    """
    _, C, _, _ = b.size()
    N, _, H, W = a.size()

    b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)
    a = a.reshape(N, -1, C, H, W).mean(1)

    return b + a


def shuffle_channels(x, groups):
    """
    Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,W] -> [N,C,H,W]
    一共C个channel要分成g组混合的channel，先把C reshape成(g, C/g)的形状，
    然后转置成(C/g, g)最后平坦成C组channel
    """
    N, C, H, W = x.size()
    x = x.reshape(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4)
    return x.reshape(N, C, H, W)


def Backbone_ResNet50_in3():
    net = resnet50(pretrained=True)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = nn.Sequential(*list(net.children())[3:5])
    div_8 = net.layer2
    div_16 = net.layer3
    div_32 = net.layer4

    return div_2, div_4, div_8, div_16, div_32


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class MINet_Res50(nn.Module):
    def __init__(self):
        super(MINet_Res50, self).__init__()
        self.div_2, self.div_4, self.div_8, self.div_16, self.div_32 = Backbone_ResNet50_in3()

        self.upsample_add = upsample_add
        self.upsample = cus_sample

        self.trans = AIM(iC_list=(64, 256, 512, 1024, 2048), oC_list=(64, 64, 64, 64, 64))

        self.sim32 = SIM(64, 32)
        self.sim16 = SIM(64, 32)
        self.sim8 = SIM(64, 32)
        self.sim4 = SIM(64, 32)
        self.sim2 = SIM(64, 32)

        self.upconv32 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data):
        in_data_2 = self.div_2(in_data)
        in_data_4 = self.div_4(in_data_2)
        in_data_8 = self.div_8(in_data_4)
        in_data_16 = self.div_16(in_data_8)
        in_data_32 = self.div_32(in_data_16)

        in_data_2, in_data_4, in_data_8, in_data_16, in_data_32 = self.trans(
            in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        )

        out_data_32 = self.upconv32(self.sim32(in_data_32))  # 1024

        out_data_16 = self.upsample_add(out_data_32, in_data_16)  # 1024
        out_data_16 = self.upconv16(self.sim16(out_data_16))

        out_data_8 = self.upsample_add(out_data_16, in_data_8)
        out_data_8 = self.upconv8(self.sim8(out_data_8))  # 512

        out_data_4 = self.upsample_add(out_data_8, in_data_4)
        out_data_4 = self.upconv4(self.sim4(out_data_4))  # 256

        out_data_2 = self.upsample_add(out_data_4, in_data_2)
        out_data_2 = self.upconv2(self.sim2(out_data_2))  # 64

        out_data_1 = self.upconv1(self.upsample(out_data_2, scale_factor=2))  # 32
        out_data = self.classifier(out_data_1)

        return out_data

if __name__ == '__main__':
    from thop import profile

    net = MINet_Res50().to('cuda:0')
    # for name, param in net.named_parameters():
    #     print(name)
    input = torch.rand(1, 3, 320, 320).to('cuda:0')
    ouput = net(input)
    flops, params = profile(net, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

# FLOPs = 87.01368G
# Params = 162.325185M


import torch
from torch import nn
from torch.nn import functional as F


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
    def __init__(self, in_ch, out_ch):
        super(AIM, self).__init__()
        # ic0, ic1, ic2, ic3, ic4 = iC_list
        # oc0, oc1, oc2, oc3, oc4 = oC_list
        self.conv0 = conv_2nV1(in_hc=in_ch, in_lc=in_ch, out_c=out_ch, main=0)
        self.conv1 = conv_3nV1(in_hc=in_ch, in_mc=in_ch, in_lc=in_ch, out_c=out_ch)
        # self.conv2 = conv_3nV1(in_hc=in_ch, in_mc=in_ch, in_lc=in_ch, out_c=out_ch)
        self.conv2 = conv_3nV1(in_hc=in_ch, in_mc=in_ch, in_lc=in_ch, out_c=out_ch)
        self.conv3 = conv_2nV1(in_hc=in_ch, in_lc=in_ch, out_c=out_ch, main=1)

    def forward(self, xs):
        # in_data_2, in_data_4, in_data_8, in_data_16, in_data_32
        out_xs = []
        out_xs.append(self.conv0(xs[0], xs[1]))
        out_xs.append(self.conv1(xs[0], xs[1], xs[2]))
        # out_xs.append(self.conv2(xs[1], xs[2], xs[3]))
        out_xs.append(self.conv2(xs[1], xs[2], xs[3]))
        out_xs.append(self.conv3(xs[2], xs[3]))

        return out_xs


class SIM(nn.Module):
    """
        自交互模块 变换-交互-融合
        高、低分辨率特征与不同通道数的交互操作，可以获得大量的不同尺度的知识
    """
    def __init__(self, h_C, l_C):
        super(SIM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)

        self.h2l_0 = nn.Conv2d(h_C, l_C, kernel_size=3, stride=1, padding=1)
        self.h2h_0 = nn.Conv2d(h_C, h_C, kernel_size=3, stride=1, padding=1)
        self.bnl_0 = nn.BatchNorm2d(l_C)
        self.bnh_0 = nn.BatchNorm2d(h_C)

        self.h2h_1 = nn.Conv2d(h_C, h_C, kernel_size=3, stride=1, padding=1)
        self.h2l_1 = nn.Conv2d(h_C, l_C, kernel_size=3, stride=1, padding=1)
        self.l2h_1 = nn.Conv2d(l_C, h_C, kernel_size=3, stride=1, padding=1)
        self.l2l_1 = nn.Conv2d(l_C, l_C, kernel_size=3, stride=1, padding=1)
        self.bnl_1 = nn.BatchNorm2d(l_C)
        self.bnh_1 = nn.BatchNorm2d(h_C)

        self.h2h_2 = nn.Conv2d(h_C, h_C, kernel_size=3, stride=1, padding=1)
        self.l2h_2 = nn.Conv2d(l_C, h_C, kernel_size=3, stride=1, padding=1)
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
        x_l2h = self.l2h_1(F.interpolate(x_l, size=(h, w), mode="bilinear"))
        x_h = self.relu(self.bnh_1(x_h2h + x_l2h))
        x_l = self.relu(self.bnl_1(x_l2l + x_h2l))

        # last conv
        x_h2h = self.h2h_2(x_h)
        x_l2h = self.l2h_2(F.interpolate(x_l, size=(h, w), mode="bilinear"))
        x_h = self.relu(self.bnh_2(x_h2h + x_l2h))

        return x_h + x


if __name__ == '__main__':
    input1 = torch.rand(1, 256, 56, 56)
    input2 = torch.rand(1, 256, 56, 56)
    net = conv_2nV1(256, 256, main=1)
    out = net(input1, input2)


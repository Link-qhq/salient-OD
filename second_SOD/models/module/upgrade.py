import torch
from torch import nn
from models.methods.ICONet import asyConv
from models.module.common import weight_init, conv3x3_bn_relu, conv3x3_bn_gelu, conv3x3_bn
from torch.nn import functional as F
from models.module.attention import CBAM, CA, ChannelAttention


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
        self.asyConv = asyConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                               padding_mode='zeros', deploy=False)
        self.oriConv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv1 = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )
        self.atrConv2 = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=3, padding=3, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )
        self.conv1 = nn.Conv2d(x, y, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(y * 4, y, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(y // 4, y),
            nn.PReLU()
        )
        self.cbam = CBAM(y, 16, 7)
        self.ca = CA(x, 16)
        self.initialize()

    def forward(self, f):
        f = f * self.ca(f)
        p1 = self.oriConv(f)  # 原始卷积
        p2 = self.asyConv(f)  # 非对称卷积
        p3 = self.atrConv1(f)  # 空洞卷积2
        p4 = self.atrConv2(f)  # 空洞卷积3
        p = torch.cat([p1, p2, p3, p4], 1)
        p = self.conv(p)
        return self.cbam(p)

    def initialize(self):
        # pass
        weight_init(self)


class FAM(nn.Module):
    """
        通道多样性特征融合
    """

    def __init__(self, in_ch, out_ch):
        super(FAM, self).__init__()
        self.dilate_conv1 = conv3x3_bn_gelu(in_ch, in_ch)
        self.dilate_conv2 = conv3x3_bn_gelu(in_ch, in_ch, dilate=2, padding=2)
        self.dilate_conv5 = conv3x3_bn_gelu(in_ch, in_ch, dilate=5, padding=5)
        self.asy_conv = asyConv(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, dilation=1,
                                groups=1,
                                padding_mode='zeros', deploy=False)

        self.conv = conv3x3_bn_gelu(in_ch * 4, in_ch)
        self.f_conv = conv3x3_bn_gelu(in_ch, out_ch)

    def forward(self, x):
        p1 = self.asy_conv(x)
        p2 = self.dilate_conv1(x)
        p3 = self.dilate_conv2(x)
        p4 = self.dilate_conv5(x)
        p = self.conv(torch.cat([p1, p2, p3, p4], dim=1))
        return self.f_conv(x + p)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.fc1(nn.Flatten()(x))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.view(b, c, h, w)


class ATM(nn.Module):
    """
        注意力模块
    """
    def __init__(self, in_ch, out_ch):
        super(ATM, self).__init__()
        self.mlp1 = Mlp(in_ch, in_ch // 8, in_ch)
        self.mlp2 = Mlp(in_ch, in_ch // 8, in_ch)
        self.conv = nn.Conv2d(in_ch * 4, in_ch, kernel_size=3, stride=1, padding=1)
        self.f_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, _, h, w = x.shape
        # 空间注意力图
        global_avgPool = torch.sigmoid(self.mlp1(F.avg_pool2d(x, (h, w))))
        global_maxPool = torch.sigmoid(self.mlp2(F.max_pool2d(x, (h, w))))
        # 通道注意力图
        maxPool = torch.sigmoid(F.max_pool2d(x, (1, 1)))
        avgPool = torch.sigmoid(F.avg_pool2d(x, (1, 1)))
        p1 = global_avgPool * maxPool
        p2 = global_avgPool * avgPool
        p3 = global_maxPool * maxPool
        p4 = global_maxPool * avgPool

        p1 = x * p1
        p2 = x * p2
        p3 = x * p3
        p4 = x * p4

        p = torch.cat([p1, p2, p3, p4], dim=1)
        return self.f_conv(self.conv(p))


class CCM(nn.Module):
    def __init__(self, infeature, out, redio):
        """ channel compression module (CCM) """
        super(CCM, self).__init__()
        self.down = nn.Conv2d(infeature, out, kernel_size=1, stride=1)
        self.channel_attention = ChannelAttention(out, redio)
        self.initialize()

    def forward(self, x):
        x = self.down(x)
        w = self.channel_attention(x)
        return x * w

    def initialize(self):
        weight_init(self)


class MultiLevelFusion(nn.Module):
    def __init__(self, in_ch):
        super(MultiLevelFusion, self).__init__()
        self.conv1 = conv3x3_bn(in_ch, in_ch)
        self.conv2 = conv3x3_bn(in_ch, in_ch)
        self.conv3 = conv3x3_bn_relu(in_ch, in_ch)
        self.conv4 = conv3x3_bn(in_ch * 2, in_ch)
        self.conv = conv3x3_bn(in_ch * 2, in_ch)
        self.ccm1 = CCM(2 * in_ch, in_ch, redio=4)
        self.ccm2 = CCM(2 * in_ch, in_ch, redio=4)
        self.down_conv = conv3x3_bn(in_ch, in_ch)
        self.up_conv = conv3x3_bn(in_ch, in_ch)
        self.initialize()

    def forward(self, low, local, high):
        low_1 = F.interpolate(low, size=local.shape[2:], mode="bilinear")
        local_1 = self.conv1(local + low_1)
        high_1 = F.interpolate(high, size=local.shape[2:], mode="bilinear")
        local_2 = self.conv2(local + high_1)
        fusion_1 = self.ccm1(torch.cat([local_1, local_2], dim=1))
        fusion_2 = self.conv3(self.down_conv(F.interpolate(low, size=local.shape[2:], mode="bilinear")) +
                              self.up_conv(F.interpolate(high, size=local.shape[2:], mode="bilinear")) + local)
        fusion = F.relu(self.ccm2(torch.cat([fusion_1, fusion_2], dim=1)), inplace=True)
        return local + fusion

    def initialize(self):
        weight_init(self)


class MultiLevelFusion1(nn.Module):
    def __init__(self, in_ch):
        super(MultiLevelFusion1, self).__init__()
        self.conv1 = conv3x3_bn_gelu(in_ch, in_ch)
        self.conv2 = conv3x3_bn_gelu(in_ch, in_ch)
        self.conv1_2 = conv3x3_bn_gelu(in_ch, in_ch)
        self.conv3 = conv3x3_bn_gelu(in_ch, in_ch)
        self.conv4 = conv3x3_bn_gelu(in_ch, in_ch)
        self.conv3_4 = conv3x3_bn_gelu(in_ch, in_ch)
        self.ccm1 = CCM(2 * in_ch, in_ch, redio=8)
        self.ccm2 = CCM(3 * in_ch, in_ch, redio=8)
        self.conv5 = conv3x3_bn_gelu(in_ch, in_ch)
        self.conv6 = conv3x3_bn_gelu(in_ch, in_ch)
        self.conv = conv3x3_bn(in_ch * 2, in_ch)
        self.initialize()

    def forward(self, low, local, high):
        low_1 = self.conv1(F.interpolate(low, size=local.shape[2:], mode="bilinear"))
        local_1 = self.conv2(local)
        fusion1 = self.conv1_2(low_1 + local_1)
        high_1 = self.conv3(F.interpolate(high, size=local.shape[2:], mode="bilinear"))
        local_2 = self.conv4(local)
        fusion2 = self.conv3_4(high_1 + local_2)
        fusion_1 = self.ccm1(torch.cat([fusion1, fusion2], dim=1))
        low_2 = self.conv5(F.interpolate(low, size=local.shape[2:], mode="bilinear"))
        high_2 = self.conv6(F.interpolate(high, size=local.shape[2:], mode="bilinear"))
        fusion_2 = self.ccm2(torch.cat([low_2, local, high_2], dim=1))
        return self.conv(torch.cat([fusion_1, fusion_2], dim=1))

    def initialize(self):
        weight_init(self)


class Contrast(nn.Module):
    def __init__(self, in_c):
        super(Contrast, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_c, in_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x

        return out  # Res


# Module-CSFI
class CSFI(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CSFI, self).__init__()
        self.l_1 = conv3x3_bn_relu(in_channel, out_channel, 1, 1)
        self.r_1 = conv3x3_bn_relu(in_channel, out_channel, 1, 1)

        self.contrast1 = Contrast(out_channel)  # Contrast模块
        self.contrast2 = Contrast(out_channel)  # Contrast模块
        self.contrast3 = Contrast(out_channel)  # Contrast模块
        self.contrast4 = Contrast(out_channel)  # Contrast模块

        self.ce_conv_l_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.ce_conv1_l_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.ce_conv1_r_1 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.ce_conv1_r_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)

        self.add_1 = conv3x3_bn_relu(out_channel, out_channel, 1, 1)
        self.add_2 = conv3x3_bn_relu(out_channel, out_channel, 1, 1)

    def forward(self, x):
        # first
        l_1 = self.l_1(x)
        r_1 = self.r_1(x)

        # second
        l_2 = self.contrast1(l_1)
        l_2 = self.ce_conv_l_1(l_2)
        r_2 = self.contrast1(r_1)
        r_2 = self.ce_conv1_r_1(r_2)

        l = self.add_1(l_2 + r_2)
        r = self.add_2()
        return None


class BIG(nn.Module):
    """
        boundary information guidance 边界信息引导模块
    """

    def __init__(self, channel, groups=8, ratio=8):
        super(BIG, self).__init__()
        self.gate1 = nn.Sequential(nn.Conv2d(channel // ratio, channel // ratio, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // ratio), nn.PReLU(),
                                   nn.Conv2d(channel // ratio, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Conv2d(channel // ratio, channel // ratio, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // ratio), nn.PReLU(),
                                   nn.Conv2d(channel // ratio, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.Conv2d(channel // ratio, channel // ratio, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // ratio), nn.PReLU(),
                                   nn.Conv2d(channel // ratio, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate4 = nn.Sequential(nn.Conv2d(channel // ratio, channel // ratio, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // ratio), nn.PReLU(),
                                   nn.Conv2d(channel // ratio, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate5 = nn.Sequential(nn.Conv2d(channel // ratio, channel // ratio, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // ratio), nn.PReLU(),
                                   nn.Conv2d(channel // ratio, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate6 = nn.Sequential(nn.Conv2d(channel // ratio, channel // ratio, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // ratio), nn.PReLU(),
                                   nn.Conv2d(channel // ratio, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate7 = nn.Sequential(nn.Conv2d(channel // ratio, channel // ratio, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // ratio), nn.PReLU(),
                                   nn.Conv2d(channel // ratio, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate8 = nn.Sequential(nn.Conv2d(channel // ratio, channel // ratio, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // ratio), nn.PReLU(),
                                   nn.Conv2d(channel // ratio, 1, kernel_size=1),
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
        x1, x2, x3, x4, x5, x6, x7, x8 = torch.split(x, self.channel // 8, dim=1)

        cm1 = self.gate1(x1)
        cm2 = self.gate2(x2)
        cm3 = self.gate3(x3)
        cm4 = self.gate4(x4)
        cm5 = self.gate5(x5)
        cm6 = self.gate6(x6)
        cm7 = self.gate7(x7)
        cm8 = self.gate8(x8)

        e1 = cm1 * torch.sigmoid(edge)
        e2 = cm2 * torch.sigmoid(edge)
        e3 = cm3 * torch.sigmoid(edge)
        e4 = cm4 * torch.sigmoid(edge)
        e5 = cm5 * torch.sigmoid(edge)
        e6 = cm6 * torch.sigmoid(edge)
        e7 = cm7 * torch.sigmoid(edge)
        e8 = cm8 * torch.sigmoid(edge)

        gv1 = F.avg_pool2d(e1, (e1.size(2), e1.size(3)), stride=(e1.size(2), e1.size(3)))
        gv2 = F.avg_pool2d(e2, (e2.size(2), e2.size(3)), stride=(e2.size(2), e2.size(3)))
        gv3 = F.avg_pool2d(e3, (e3.size(2), e3.size(3)), stride=(e3.size(2), e3.size(3)))
        gv4 = F.avg_pool2d(e4, (e4.size(2), e4.size(3)), stride=(e4.size(2), e4.size(3)))
        gv5 = F.avg_pool2d(e5, (e5.size(2), e5.size(3)), stride=(e5.size(2), e5.size(3)))
        gv6 = F.avg_pool2d(e6, (e6.size(2), e6.size(3)), stride=(e6.size(2), e6.size(3)))
        gv7 = F.avg_pool2d(e7, (e7.size(2), e7.size(3)), stride=(e7.size(2), e7.size(3)))
        gv8 = F.avg_pool2d(e8, (e8.size(2), e8.size(3)), stride=(e8.size(2), e8.size(3)))
        weight = self.weight(torch.cat((gv1, gv2, gv3, gv4, gv5, gv6, gv7, gv8), 1))
        w1, w2, w3, w4, w5, w6, w7, w8 = torch.split(weight, 1, dim=1)

        nx1 = x1 * w1
        nx2 = x2 * w2
        nx3 = x3 * w3
        nx4 = x4 * w4
        nx5 = x5 * w5
        nx6 = x6 * w6
        nx7 = x7 * w7
        nx8 = x8 * w8

        return self.conv(torch.cat((nx1, nx2, nx3, nx4, nx5, nx6, nx7, nx8), 1))

if __name__ == '__main__':
    a = torch.rand(1, 64, 56, 56)
    edge = torch.rand(1, 1, 56, 56)
    net = BIG(64)
    print(net(a, edge).shape)
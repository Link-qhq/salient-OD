import math

import torch
from torch import nn
import torch.nn.functional as F

from models.module.attention import SpatialAttention
from models.module.common import conv3x3_bn_relu, con1x1_bn, conv3x3_bn_pRelu
from models.backbone.Transformer import Block
from timm.models.res2net import Bottle2neck


# multi-scale feature aggregation module
class MSFAM(nn.Module):
    def __init__(self, ch1=256, ch2=512, ch3=1024, ch4=2048):
        super(MSFAM, self).__init__()
        self.conv1_1 = conv3x3_bn_pRelu(ch1, ch1 // 2, 1, 1)
        self.conv2_1 = conv3x3_bn_pRelu(ch2, ch2 // 2, 1, 1)
        self.conv3_1 = conv3x3_bn_pRelu(ch3, ch3 // 2, 1, 1)
        self.conv4_1 = conv3x3_bn_pRelu(ch4, ch4 // 2, 1, 1)

        self.conv2_2 = conv3x3_bn_pRelu(ch1, ch1 // 2, 1, 1)
        self.conv3_2 = conv3x3_bn_pRelu(ch2, ch2 // 2, 1, 1)
        self.conv4_2 = conv3x3_bn_pRelu(ch3, ch3 // 2, 1, 1)

        self.conv3_3 = conv3x3_bn_pRelu(ch1, ch1 // 2, 1, 1)
        self.conv4_3 = conv3x3_bn_pRelu(ch2, ch2 // 2, 1, 1)

        self.conv4_4 = conv3x3_bn_pRelu(ch1, ch1 // 2, 1, 1)

        self.down1_1 = con1x1_bn(ch1 // 2, ch1, 2)
        self.down2_1 = con1x1_bn(ch2 // 2, ch2, 2)
        self.down3_1 = con1x1_bn(ch3 // 2, ch3, 2)

        self.down2_2 = con1x1_bn(ch1 // 2, ch1, 2)
        self.down3_2 = con1x1_bn(ch2 // 2, ch2, 2)

        self.down3_3 = con1x1_bn(ch1 // 2, ch1, 2)

        self.conv1 = nn.Conv2d(ch1 // 2, 64, 1, 1)
        self.conv2 = nn.Conv2d(ch1 // 2, 64, 1, 1)
        self.conv3 = nn.Conv2d(ch1 // 2, 64, 1, 1)
        self.conv4 = nn.Conv2d(ch1 // 2, 64, 1, 1)

    def forward(self, fea1, fea2, fea3, fea4):
        out1_1 = self.conv1_1(fea1)  # 128
        out2_1 = self.conv2_1(fea2)  # 256
        out3_1 = self.conv3_1(fea3)  # 512
        out4_1 = self.conv4_1(fea4)  # 1024

        tmp1_1 = self.down1_1(out1_1)
        tmp2_1 = self.down2_1(out2_1)
        tmp3_1 = self.down3_1(out3_1)

        out2_2 = self.conv2_2(tmp1_1 + out2_1)
        out3_2 = self.conv3_2(tmp2_1 + out3_1)
        out4_2 = self.conv4_2(tmp3_1 + out4_1)

        tmp2_2 = self.down2_2(out2_2)
        tmp3_2 = self.down3_2(out3_2)

        out3_3 = self.conv3_3(tmp2_2 + out3_2)
        out4_3 = self.conv4_3(tmp3_2 + out4_2)

        tmp3_3 = self.down3_3(out3_3)
        out4_4 = self.conv4_4(tmp3_3 + out4_3)

        out1 = self.conv1(out1_1)
        out2 = self.conv2(out2_2)
        out3 = self.conv3(out3_3)
        out4 = self.conv4(out4_4)
        return out1, out2, out3, out4


# cross feature refinement module
class CFRM(nn.Module):
    def __init__(self, channel=64):  # 适用64
        super(CFRM, self).__init__()
        self.conv_left_1 = conv3x3_bn_relu(channel, channel)
        self.conv_left_2 = conv3x3_bn_relu(channel, channel)

        self.conv_down_1 = conv3x3_bn_relu(channel, channel)
        self.conv_down_2 = conv3x3_bn_relu(channel, channel)
        self.tf = Block(channel, 2, 3)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channel, channel, 7, 1, 3),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.up = conv3x3_bn_relu(channel, channel)
        self.right = conv3x3_bn_relu(channel, channel)

    def forward(self, left, down):
        if down.shape[2:] != left.shape[2:]:
            down = F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=True)
        left_1 = self.conv_left_1(left)
        left_2 = self.conv_left_2(left_1)

        down_1 = self.conv_down_1(down)
        down_2 = self.conv_down_2(down_1)

        fuse = left_2 * down_2
        # coe = F.sigmoid(self.conv_fuse(
        #     torch.cat([
        #         F.max_pool2d(fuse, 7, 1, 3),
        #         F.avg_pool2d(fuse, 7, 1, 3)
        #     ], 1)
        # ))
        coe = self.tf(fuse)
        up = self.right(left_1 + coe)
        right = self.up(down_1 + coe)
        return up, right


# Filtering and Fusion Module
class FFM(nn.Module):
    def __init__(self, ch_shadow, ch_deep, embed_dim=384, mlp_ratio=3, num_heads=1):
        super(FFM, self).__init__()
        self.conv_up = nn.Conv2d(ch_deep, ch_shadow, 1, 1)
        self.conv_fds = conv3x3_bn_relu(ch_shadow, ch_shadow, 3, 1, 1)
        # self.tf = TokenFusion()
        self.tf = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
        # self.tf = MixedAttention(in_dim=in_dim, dim=embed_dim, img_size=img_size,
        #                          window_size=window_size, num_heads=num_heads, mlp_ratio=mlp_ratio, depth=1)
        self.conv_fuse = conv3x3_bn_relu(ch_shadow, ch_shadow, 1, 1)

    def forward(self, shadow, deep):
        B, C, H, W = shadow.shape
        deep = F.interpolate(self.conv_up(deep), size=shadow.shape[2:], mode='bilinear', align_corners=True)
        Fds = self.tf((deep + shadow).reshape(B, H * W, C)).reshape(B, C, H, W)
        weight = F.sigmoid(self.conv_fuse(Fds))
        fusion = self.conv_fuse(weight * deep + weight * shadow)
        return fusion, fusion



class Mlp(nn.Module):
    def __init__(self, inc, hidden=None, outc=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        outc = outc or inc
        hidden = hidden or inc
        self.fc1 = nn.Linear(inc, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, outc)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# Long-range dependency module
class LRDM(nn.Module):
    def __init__(self, channel):
        super(LRDM, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # 1, h, w
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // 2),
            nn.GELU(),
            nn.Linear(channel // 2, channel),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        aap = self.aap(x)
        Bs, C, H, W = aap.shape
        mlp = self.mlp(aap.reshape(Bs, H * W, C))
        return x * torch.sigmoid(mlp.reshape(Bs, C, H, W))


# Edge refinement module
class ERM(nn.Module):
    def __init__(self):
        super(ERM, self).__init__()
        self.conv1 = Bottle2neck()
        self.conv2 = Bottle2neck()

    def forward(self, shadow, coarse):
        return self.conv2(self.conv1(shadow) + coarse)


# https://github.com/sunny2109/SAFMN
# 论文：https://arxiv.org/pdf/2302.13800
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


# Convolutional Channel Mixer
class CCM(nn.Module):
    def __init__(self, dim, ffn_scale, use_se=False):
        super().__init__()
        self.use_se = use_se
        hidden_dim = int(dim*ffn_scale)

        self.conv1 = nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x


class CSP(nn.Module):
    def __init__(self, channel):
        super(CSP, self).__init__()
        self.conv_1 = conv3x3_bn_pRelu(channel, channel)
        self.conv_2 = conv3x3_bn_pRelu(channel, channel)
        self.conv_3 = conv3x3_bn_pRelu(channel, channel)
        self.conv_4 = conv3x3_bn_pRelu(channel, channel)

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.conv_4(torch.cat([x2, x3], 1))
        out = torch.cat([x1, x4], 1)
        return out


def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k


# Temporal Fusion Attention Module
# Exchanging Dual-Encoder–Decoder: A New Strategy for Change Detection With Semantic Guidance and Spatial Localization
class TFAM(nn.Module):
    """Fuse two feature into one feature."""

    def __init__(self, in_channel):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.k = kernel_size(in_channel)
        self.channel_conv1 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.channel_conv2 = nn.Conv1d(4, 1, kernel_size=self.k, padding=self.k // 2)
        self.spatial_conv1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.spatial_conv2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
        self.softmax = nn.Softmax(0)

    def forward(self, t1, t2, log=None, module_name=None,
                img_name=None):
        # channel part
        t1_channel_avg_pool = self.avg_pool(t1)  # b,c,1,1
        t1_channel_max_pool = self.max_pool(t1)  # b,c,1,1
        t2_channel_avg_pool = self.avg_pool(t2)  # b,c,1,1
        t2_channel_max_pool = self.max_pool(t2)  # b,c,1,1

        channel_pool = torch.cat([t1_channel_avg_pool, t1_channel_max_pool,
                                  t2_channel_avg_pool, t2_channel_max_pool],
                                 dim=2).squeeze(-1).transpose(1, 2)  # b,4,c
        t1_channel_attention = self.channel_conv1(channel_pool)  # b,1,c
        t2_channel_attention = self.channel_conv2(channel_pool)  # b,1,c
        channel_stack = torch.stack([t1_channel_attention, t2_channel_attention],
                                    dim=0)  # 2,b,1,c
        channel_stack = self.softmax(channel_stack).transpose(-1, -2).unsqueeze(-1)  # 2,b,c,1,1

        # spatial part
        t1_spatial_avg_pool = torch.mean(t1, dim=1, keepdim=True)  # b,1,h,w
        t1_spatial_max_pool = torch.max(t1, dim=1, keepdim=True)[0]  # b,1,h,w
        t2_spatial_avg_pool = torch.mean(t2, dim=1, keepdim=True)  # b,1,h,w
        t2_spatial_max_pool = torch.max(t2, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_pool = torch.cat([t1_spatial_avg_pool, t1_spatial_max_pool,
                                  t2_spatial_avg_pool, t2_spatial_max_pool], dim=1)  # b,4,h,w
        t1_spatial_attention = self.spatial_conv1(spatial_pool)  # b,1,h,w
        t2_spatial_attention = self.spatial_conv2(spatial_pool)  # b,1,h,w
        spatial_stack = torch.stack([t1_spatial_attention, t2_spatial_attention], dim=0)  # 2,b,1,h,w
        spatial_stack = self.softmax(spatial_stack)  # 2,b,1,h,w

        # fusion part, add 1 means residual add
        stack_attention = channel_stack + spatial_stack + 1  # 2,b,c,h,w
        fuse = stack_attention[0] * t1 + stack_attention[1] * t2  # b,c,h,w

        return fuse


class MBIG(nn.Module):  # modify BIG
    def __init__(self, in_ch, ratio=4):
        super(MBIG, self).__init__()
        inter_channels = in_ch // ratio
        self.spatial_att = SpatialAttention()
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
        self.local_att1 = nn.Sequential(
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
        # 局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(in_ch, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, in_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_ch)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_channels, 1, 1),
            nn.BatchNorm2d(inter_channels),
            nn.PReLU(),
            nn.Conv2d(inter_channels, in_ch, 1, 1),
        )

    def forward(self, x, edge):
        weight = torch.sigmoid(edge)
        spatial_coe = self.spatial_att(x)
        boundary = (spatial_coe + weight) * x
        boundary = boundary * torch.sigmoid(self.local_att1(boundary) + self.global_att1(boundary))
        region = (spatial_coe + (1. - weight)) * x
        region = region * torch.sigmoid(self.local_att2(region) + self.global_att2(region))
        return self.fuse_conv(boundary + region)


if __name__ == '__main__':
    f1 = torch.rand(2, 64, 64, 64)
    f2 = torch.rand(2, 1, 64, 64)
    f3 = torch.rand(1, 1024, 16, 16)
    f4 = torch.rand(1, 2048, 8, 8)
    model = MBIG(64)
    out = model(f1, f2)
    print(out.shape)
    # model = MSFAM()
    # model = LRDM(512)
    # model = SAFM(512)
    # out = model(f2)
    # left = torch.rand(1, 64, 64, 64)
    # down = torch.rand(1, 64, 32, 32)
    # model = FFM(64, 64, 64)
    # out = model(left, down)
    # model = TFAM(in_channel=64)
    # flops, params = profile(model, inputs=(torch.randn(1, 64, 32, 32), torch.randn(1, 64, 32, 32)))
    # print(f"FLOPs: {flops}, Params: {params}")

import math
from functools import partial
import torch
from timm.layers import DropPath, trunc_normal_
from torch import nn
from models.base_model import Net
from models.module.common import conv3x3_bn_relu, conv1x1_bn_relu
import torch.nn.functional as F
from models.module.attention import SpatialAttention, AttentionBlock, \
    SpaceAttention, CBAM
from models.backbone.PVT_V2 import Attention, Mlp


class MFIM(nn.Module):
    def __init__(self, channel):
        super(MFIM, self).__init__()
        self.channel_single = int(channel // 2)
        self.self_conv = conv1x1_bn_relu(channel, channel)
        self.pre = conv1x1_bn_relu(channel, self.channel_single)
        self.ori_conv = conv3x3_bn_relu(channel, self.channel_single)
        self.atr2_conv = nn.Sequential(
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            conv1x1_bn_relu(self.channel_single, self.channel_single),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True)
        )
        self.atr3_conv = nn.Sequential(
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(1, 5), stride=1, padding=(0, 2)),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(5, 1), stride=1, padding=(2, 0)),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            conv1x1_bn_relu(self.channel_single, self.channel_single),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True)
        )
        self.atr5_conv = nn.Sequential(
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            conv1x1_bn_relu(self.channel_single, self.channel_single),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True)
        )
        self.pool_conv = nn.Sequential(
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(1, 9), stride=1, padding=(0, 4)),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(9, 1), stride=1, padding=(4, 0)),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            conv1x1_bn_relu(self.channel_single, self.channel_single),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.channel_single), nn.ReLU(True)
        )
        self.fuse_atr2 = conv3x3_bn_relu(channel, self.channel_single)
        self.fuse_atr3 = conv3x3_bn_relu(channel, self.channel_single)
        self.fuse_atr4 = conv3x3_bn_relu(channel, self.channel_single)
        self.fuse_atr5 = conv3x3_bn_relu(channel, self.channel_single)
        self.fuse_conv = conv3x3_bn_relu(self.channel_single * 5, channel)

    def forward(self, x):
        clone = x
        I1 = self.ori_conv(x)

        x = self.pre(x)

        I2 = self.atr2_conv(x)
        I2 = self.fuse_atr2(torch.cat([I1, I2], 1))

        I3 = self.atr3_conv(x)
        I3 = self.fuse_atr3(torch.cat([I2, I3], 1))

        I4 = self.atr5_conv(x)
        I4 = self.fuse_atr4(torch.cat([I3, I4], 1))

        I5 = self.pool_conv(x)
        I5 = self.fuse_atr5(torch.cat([I4, I5], 1))
        return self.self_conv(clone) + self.fuse_conv(torch.cat([I1, I2, I3, I4, I5], 1))


# CA 注意力机制
class LocalAttention(nn.Module):
    def __init__(self, channel=64, kernel_size=7):
        super(LocalAttention, self).__init__()
        pad = kernel_size // 2
        self.conv_h = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size, padding=pad, groups=channel, bias=False),
            nn.GroupNorm(16, channel),
            nn.Sigmoid()
        )
        self.conv_w = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size, padding=pad, groups=channel, bias=False),
            nn.GroupNorm(16, channel),
            nn.Sigmoid()
        )

    def forward(self, x, H, W):
        b, _, c = x.shape
        x = x.view(b, c, H, W)
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, H)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, W)
        x_h = self.conv_h(x_h).view(b, c, H, 1)
        x_w = self.conv_w(x_w).view(b, c, 1, W)
        return (x_h * x_w).view(b, H * W, c)


# 设计灵感来自于卷积神经网络和transform的互补性,它们分别擅长于提取局部特征和全局特征
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        self.attn2 = LocalAttention()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn1(self.norm1(x), H, W) + self.attn2(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Decoder(nn.Module):
    def __init__(self, in_channel_list, img_size=224, channel=64):
        super(Decoder, self).__init__()
        set_channels = channel
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # CNN feature channel compression
        self.down1 = conv3x3_bn_relu(in_channel_list[0], set_channels)
        self.down2 = conv3x3_bn_relu(in_channel_list[1], set_channels)
        self.down3 = conv3x3_bn_relu(in_channel_list[2], set_channels)
        self.down4 = conv3x3_bn_relu(in_channel_list[3], set_channels)
        # semantic enhancement module
        self.se_att1 = AttentionBlock(set_channels)
        self.se_att2 = AttentionBlock(set_channels)
        self.se_att3 = AttentionBlock(set_channels)
        self.se_att4 = AttentionBlock(set_channels)
        # spatial enhancement module
        depths = [1, 1, 1, 1]
        num_stages = 4
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.pe_att1 = Block(dim=set_channels,
                             num_heads=8,
                             mlp_ratio=4,
                             qkv_bias=True,
                             drop_path=dpr[cur],
                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             linear=True,
                             sr_ratio=8)
        cur += depths[0]
        self.pe_att2 = Block(dim=set_channels,
                             num_heads=8,
                             mlp_ratio=4,
                             qkv_bias=True,
                             drop_path=dpr[cur],
                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             linear=True,
                             sr_ratio=4)
        cur += depths[1]
        self.pe_att3 = Block(dim=set_channels,
                             num_heads=8,
                             mlp_ratio=4,
                             qkv_bias=True,
                             drop_path=dpr[cur],
                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             linear=True,
                             sr_ratio=2)
        cur += depths[2]
        self.pe_att4 = Block(dim=set_channels,
                             num_heads=8,
                             mlp_ratio=4,
                             qkv_bias=True,
                             drop_path=dpr[cur],
                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             linear=True,
                             sr_ratio=1)
        cur += depths[3]
        self.dce1 = MFIM(set_channels)
        self.dce2 = MFIM(set_channels)
        self.dce3 = MFIM(set_channels)
        self.dce4 = MFIM(set_channels)
        # self.dce5 = MFIM(set_channels)

        # channel reduction
        self.cr43 = conv3x3_bn_relu(set_channels * 2, set_channels)
        self.cr32 = conv3x3_bn_relu(set_channels * 2, set_channels)
        self.cr21 = conv3x3_bn_relu(set_channels * 2, set_channels)
        self.cr10 = conv3x3_bn_relu(set_channels * 2, set_channels)

        self.conv_1 = nn.Conv2d(set_channels, 1, 3, 1, 1)
        self.conv_2 = nn.Conv2d(set_channels, 1, 3, 1, 1)
        self.conv_3 = nn.Conv2d(set_channels, 1, 3, 1, 1)
        self.conv_4 = nn.Conv2d(set_channels, 1, 3, 1, 1)

    def forward(self, fea, shape):
        # 1/4, 1/8, 1/16, 1/32
        # 256, 512, 1024, 2048
        fea1 = self.down1(fea[0])
        fea2 = self.down2(fea[1])
        fea3 = self.down3(fea[2])
        fea4 = self.down4(fea[3])
        se1 = self.se_att1(fea1)
        se2 = self.se_att2(fea2)
        se3 = self.se_att3(fea3)
        se4 = self.se_att4(fea4)

        dce4 = self.dce4(se4)  # 64
        B, _, H, W = dce4.shape
        pe4_map = self.up(self.pe_att4(dce4.reshape(B, H * W, _), H, W).reshape(B, _, H, W))
        pe3 = se3 * pe4_map
        up43 = self.up(dce4)
        cr43 = self.cr43(torch.cat([pe3, up43], 1))

        dce3 = self.dce3(cr43)
        B, _, H, W = dce3.shape
        pe3_map = self.up(self.pe_att3(dce3.reshape(B, H * W, _), H, W).reshape(B, _, H, W))
        pe2 = se2 * pe3_map
        up32 = self.up(dce3)
        cr32 = self.cr32(torch.cat([pe2, up32], 1))

        dce2 = self.dce2(cr32)
        B, _, H, W = dce2.shape
        pe2_map = self.up(self.pe_att2(dce2.reshape(B, H * W, _), H, W).reshape(B, _, H, W))
        pe1 = se1 * pe2_map
        up21 = self.up(dce2)
        cr21 = self.cr21(torch.cat([pe1, up21], 1))
        dce1 = self.dce1(cr21)

        dce1 = F.interpolate(self.conv_1(dce1), size=shape, mode='bilinear', align_corners=True)
        dce2 = F.interpolate(self.conv_2(dce2), size=shape, mode='bilinear', align_corners=True)
        dce3 = F.interpolate(self.conv_3(dce3), size=shape, mode='bilinear', align_corners=True)
        dce4 = F.interpolate(self.conv_4(dce4), size=shape, mode='bilinear', align_corners=True)
        return dce1, dce2, dce3, dce4


if __name__ == '__main__':
    from thop import profile

    input = torch.rand(2, 3, 320, 320)
    edge = torch.rand(2, 1, 56, 56)
    model = Net(backbone='pvt', decoder=Decoder)
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
# FLOPs = 23.12435648G
# Params = 27.104196M
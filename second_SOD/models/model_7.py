import torch
from torch import nn
from models.base_model import Net
from models.module.common import conv3x3_bn_relu, conv1x1_bn_relu
import torch.nn.functional as F
from models.module.attention import SpatialAttention, AttentionBlock, \
    SpaceAttention, CBAM
from models.backbone.PVT_V2 import Attention


class MFIM(nn.Module):
    def __init__(self, channel):
        super(MFIM, self).__init__()
        self.channel_single = int(channel // 2)
        self.self_conv = conv1x1_bn_relu(channel, channel)
        self.pre = conv1x1_bn_relu(channel, self.channel_single)
        self.ori_conv = conv3x3_bn_relu(channel, self.channel_single)
        self.atr2_conv = nn.Sequential(
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(1, 3), stride=1, padding=(0, 1)), nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(3, 1), stride=1, padding=(1, 0)), nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            conv1x1_bn_relu(self.channel_single, self.channel_single),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=3, stride=1, padding=1, dilation=1), nn.BatchNorm2d(self.channel_single), nn.ReLU(True)
        )
        self.atr3_conv = nn.Sequential(
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(1, 5), stride=1, padding=(0, 2)), nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(5, 1), stride=1, padding=(2, 0)), nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            conv1x1_bn_relu(self.channel_single, self.channel_single),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=3, stride=1, padding=2, dilation=2), nn.BatchNorm2d(self.channel_single), nn.ReLU(True)
        )
        self.atr5_conv = nn.Sequential(
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(1, 7), stride=1, padding=(0, 3)), nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(7, 1), stride=1, padding=(3, 0)), nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            conv1x1_bn_relu(self.channel_single, self.channel_single),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=3, stride=1, padding=3, dilation=3), nn.BatchNorm2d(self.channel_single), nn.ReLU(True)
        )
        self.pool_conv = nn.Sequential(
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(1, 9), stride=1, padding=(0, 4)), nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=(9, 1), stride=1, padding=(4, 0)), nn.BatchNorm2d(self.channel_single), nn.ReLU(True),
            conv1x1_bn_relu(self.channel_single, self.channel_single),
            nn.Conv2d(self.channel_single, self.channel_single, kernel_size=3, stride=1, padding=5, dilation=5), nn.BatchNorm2d(self.channel_single), nn.ReLU(True)
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


class DCE_Module(nn.Module):
    def __init__(self, input_channels):
        super(DCE_Module, self).__init__()
        self.input_channels = input_channels
        self.concat_channels = int(input_channels * 2)
        self.channels_single = int(input_channels / 4)
        self.channels_double = int(input_channels / 2)

        self.local_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.input_channels), nn.ReLU(True))

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))

        self.p1_d1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, (3, 1), 1, padding=(1, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p1_d2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, (1, 3), 1, padding=(0, 1)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True),
            nn.Conv2d(self.channels_single, self.channels_single, (3, 1), 1, padding=(1, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p1_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 1, 1, 0),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))

        self.p2_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (5, 1), 1, padding=(2, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 5), 1, padding=(0, 2)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p2_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 5), 1, padding=(0, 2)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True),
            nn.Conv2d(self.channels_single, self.channels_single, (5, 1), 1, padding=(2, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p2_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 1, 1, 0),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))

        self.p3_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (7, 1), 1, padding=(3, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 7), 1, padding=(0, 3)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p3_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 7), 1, padding=(0, 3)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True),
            nn.Conv2d(self.channels_single, self.channels_single, (7, 1), 1, padding=(3, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p3_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 1, 1, 0),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))

        self.p4_d1 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (9, 1), 1, padding=(4, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True),
            nn.Conv2d(self.channels_single, self.channels_single, (1, 9), 1, padding=(0, 4)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p4_d2 = nn.Sequential(
            nn.Conv2d(self.channels_double, self.channels_single, (1, 9), 1, padding=(0, 4)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True),
            nn.Conv2d(self.channels_single, self.channels_single, (9, 1), 1, padding=(4, 0)),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p4_fusion = nn.Sequential(nn.Conv2d(self.channels_double, self.channels_single, 1, 1, 0),
                                       nn.BatchNorm2d(self.channels_single), nn.ReLU(True))
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=9, dilation=9),
            nn.BatchNorm2d(self.channels_single), nn.ReLU(True))

        self.fusion = nn.Sequential(nn.Conv2d(self.concat_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU(True))

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1_fusion(torch.cat((self.p1_d1(p1_input), self.p1_d2(p1_input)), 1))
        p1 = self.p1_dc(p1)

        p2_input = torch.cat((self.p2_channel_reduction(x), p1), 1)
        #
        p2 = self.p2_fusion(torch.cat((self.p2_d1(p2_input), self.p2_d2(p2_input)), 1))
        p2 = self.p2_dc(p2)

        p3_input = torch.cat((self.p3_channel_reduction(x), p2), 1)
        p3 = self.p3_fusion(torch.cat((self.p3_d1(p3_input), self.p3_d2(p3_input)), 1))
        p3 = self.p3_dc(p3)

        p4_input = torch.cat((self.p4_channel_reduction(x), p3), 1)
        p4 = self.p4_fusion(torch.cat((self.p4_d1(p4_input), self.p4_d2(p4_input)), 1))
        p4 = self.p4_dc(p4)

        local_conv = self.local_conv(x)

        dce = self.fusion(torch.cat((p1, p2, p3, p4, local_conv), 1))

        return dce


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
        # self.down1 = RFB(256, set_channels)
        # self.down2 = RFB(512, set_channels)
        # self.down3 = RFB(1024, set_channels)
        # self.down4 = RFB(2048, set_channels)
        # semantic enhancement module
        self.se_att1 = AttentionBlock(set_channels)
        self.se_att2 = AttentionBlock(set_channels)
        self.se_att3 = AttentionBlock(set_channels)
        self.se_att4 = AttentionBlock(set_channels)
        # spatial enhancement module
        # self.pe_att1 = SpatialAttention()
        # self.pe_att2 = SpatialAttention()
        # self.pe_att3 = SpatialAttention()
        # self.pe_att4 = SpatialAttention()

        self.pe_att1 = Attention(dim=set_channels,
                                 num_heads=8,
                                 qkv_bias=True,
                                 linear=True,
                                 sr_ratio=8)
        self.pe_att2 = Attention(dim=set_channels,
                                 num_heads=8,
                                 qkv_bias=True,
                                 linear=True,
                                 sr_ratio=4)
        self.pe_att3 = Attention(dim=set_channels,
                                 num_heads=8,
                                 qkv_bias=True,
                                 linear=True,
                                 sr_ratio=2)
        self.pe_att4 = Attention(dim=set_channels,
                                 num_heads=8,
                                 qkv_bias=True,
                                 linear=True,
                                 sr_ratio=1)

        # self.dce1 = DCE_Module(set_channels)
        # self.dce2 = DCE_Module(set_channels)
        # self.dce3 = DCE_Module(set_channels)
        # self.dce4 = DCE_Module(set_channels)
        self.dce1 = MFIM(set_channels)
        self.dce2 = MFIM(set_channels)
        self.dce3 = MFIM(set_channels)
        self.dce4 = MFIM(set_channels)

        # channel reduction
        self.cr43 = conv3x3_bn_relu(set_channels * 2, set_channels)
        self.cr32 = conv3x3_bn_relu(set_channels * 2, set_channels)
        self.cr21 = conv3x3_bn_relu(set_channels * 2, set_channels)
        self.cr10 = conv3x3_bn_relu(set_channels * 2, set_channels)

        # up
        # self.up43 = nn.Sequential(conv3x3_bn_relu(set_channels, set_channels), self.up)
        # self.up32 = nn.Sequential(conv3x3_bn_relu(set_channels, set_channels), self.up)
        # self.up21 = nn.Sequential(conv3x3_bn_relu(set_channels, set_channels), self.up)
        # self.up10 = nn.Sequential(conv3x3_bn_relu(set_channels, set_channels), self.up)

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

    input = torch.rand(2, 3, 224, 224)
    edge = torch.rand(2, 1, 56, 56)
    model = Net(backbone='pvt', decoder=Decoder)
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

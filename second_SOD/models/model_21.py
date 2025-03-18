import torch
from torch import nn
from models.module.attention import ChannelAttention, SpatialAttention, SegNext_Attention
from models.module.common import conv3x3_bn_pRelu
from models.backbone.resnet import ResNet
import torch.nn.functional as F


# feature Fusion module
class FFM(nn.Module):
    def __init__(self, in_left, in_down):  # left down
        super(FFM, self).__init__()
        self.up_conv = conv3x3_bn_pRelu(in_down, in_left, 1, 1)
        self.ca_att = ChannelAttention(in_left)
        self.sa_att = SpatialAttention()
        self.conv_fuse = conv3x3_bn_pRelu(in_left * 2, in_left, 1, 1)
        self.conv_left = conv3x3_bn_pRelu(in_left * 2, in_left, 1, 1)
        self.conv_down = conv3x3_bn_pRelu(in_left * 2, in_left, 1, 1)
        self.conv_final = conv3x3_bn_pRelu(in_left * 2, in_left, 1, 1)
        self.att = SegNext_Attention(in_left)

    def forward(self, left, down):
        B, _, H, W = left.shape
        if down.shape[2:] != left.shape[2:]:
            down = self.up_conv(F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=False))
        sa = self.sa_att(down)
        ca = self.ca_att(left)
        fuse = self.conv_fuse(torch.cat([sa * left, ca * down], 1))
        # fuse = self.att(fuse.reshape(B, H * W, _)).reshape(B, _, H, W)
        # fuse = self.att(fuse)
        left = self.conv_left(torch.cat([fuse, left], 1))
        down = self.conv_down(torch.cat([fuse, down], 1))
        out = self.conv_final(torch.cat([left, down], 1))
        out = self.att(out)
        return out, out


class AddFuse(nn.Module):
    def __init__(self, channel=64):
        super(AddFuse, self).__init__()
        # self.conv = conv3x3_bn_pRelu(channel, channel, 1, 1)

    def forward(self, up, fea, extra=None):
        if up.shape[2:] != fea.shape[2:]:
            up = F.interpolate(up, size=fea.shape[2:], mode='bilinear', align_corners=True)
        if extra is not None and extra.shape[2:] != fea.shape[2:]:
            extra = F.interpolate(extra, size=fea.shape[2:], mode='bilinear', align_corners=True)
        return up + fea + extra if extra is not None else up + fea


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.left_conv_1 = conv3x3_bn_pRelu(channel, channel)
        self.left_conv_2 = conv3x3_bn_pRelu(channel, channel)
        self.left_conv_3 = conv3x3_bn_pRelu(channel, channel)
        self.left_conv_4 = conv3x3_bn_pRelu(channel, channel)
        self.down_conv_1 = conv3x3_bn_pRelu(channel, channel)
        self.down_conv_2 = conv3x3_bn_pRelu(channel, channel)
        self.down_conv_3 = conv3x3_bn_pRelu(channel, channel)
        self.down_conv_4 = conv3x3_bn_pRelu(channel, channel)

    def forward(self, left, down):
        left_1 = self.left_conv_1(left)
        down_1 = self.down_conv_1(down)
        left_2 = self.left_conv_2(left_1)
        down_2 = self.down_conv_2(down_1)

        left_down = F.interpolate(down_2, size=left.shape[2:], mode='bilinear', align_corners=True)
        down_up = F.interpolate(left_2, size=down.shape[2:], mode='bilinear', align_corners=True)

        left_3 = self.left_conv_3(left_2 + left_down)
        down_3 = self.down_conv_3(down_2 + down_up)

        left_4 = self.left_conv_4(left_1 + left_3)
        down_4 = self.down_conv_4(down_1 + down_3)
        return left_4, down_4


class SubEncoder(nn.Module):
    def __init__(self, channel):
        super(SubEncoder, self).__init__()
        self.cfm1 = CFM(channel)
        self.cfm2 = CFM(channel)
        self.cfm3 = CFM(channel)

        self.cfm4 = CFM(channel)
        self.cfm5 = CFM(channel)
        self.cfm6 = CFM(channel)

    def forward(self, fea1, fea2, fea3, fea4):
        up3, right3 = self.cfm3(fea3, fea4)
        up2, right2 = self.cfm2(fea2, up3)
        up1, right1 = self.cfm1(fea1, up2)

        up4, right4 = self.cfm4(up1, right1)
        up5, right5 = self.cfm5(right4, right2)
        up6, right6 = self.cfm6(right5, right3)

        return up1, up4, up5, up6, right6


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        set_channels = 64
        # self.msfam = MSFAM()
        self.down1 = conv3x3_bn_pRelu(256, set_channels)
        self.down2 = conv3x3_bn_pRelu(512, set_channels)
        self.down3 = conv3x3_bn_pRelu(1024, set_channels)
        self.down4 = conv3x3_bn_pRelu(2048, set_channels)
        self.sub1 = SubEncoder(set_channels)
        self.sub2 = SubEncoder(set_channels)

        # self.dwf1 = DWF(set_channels, 0)
        # self.dwf2 = DWF(set_channels, 1)
        # self.dwf3 = DWF(set_channels, 2)
        # self.dwf4 = DWF(set_channels, 3)

        self.conv_up1 = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_up2 = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_right1 = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_right2 = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_right3 = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_right4 = nn.Conv2d(set_channels, 1, 1, 1)

    def forward(self, fea, shape):
        fea1 = self.down1(fea[0])
        fea2 = self.down2(fea[1])
        fea3 = self.down3(fea[2])
        fea4 = self.down4(fea[3])

        up1, right1, right2, right3, right4 = self.sub1(fea1, fea2, fea3, fea4)
        up2, right1, right2, right3, right4 = self.sub2(right1, right2, right3, right4)

        up1 = F.interpolate(self.conv_up1(up1), size=shape, mode='bilinear', align_corners=True)
        up2 = F.interpolate(self.conv_up2(up2), size=shape, mode='bilinear', align_corners=True)

        # right1 = self.dwf1((right1, right2, right3, right4))
        # right2 = self.dwf2((right1, right2, right3, right4))
        # right3 = self.dwf3((right1, right2, right3, right4))
        # right4 = self.dwf4((right1, right2, right3, right4))
        right1 = F.interpolate(self.conv_right1(right1), size=shape, mode='bilinear', align_corners=True)
        right2 = F.interpolate(self.conv_right2(right2), size=shape, mode='bilinear', align_corners=True)
        right3 = F.interpolate(self.conv_right3(right3), size=shape, mode='bilinear', align_corners=True)
        right4 = F.interpolate(self.conv_right4(right4), size=shape, mode='bilinear', align_corners=True)
        return up1, up2, right1, right2, right3, right4


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

    input = torch.rand(2, 3, 288, 288)
    edge = torch.rand(2, 1, 56, 56)
    model = PGN()
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

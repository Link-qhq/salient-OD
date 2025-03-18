import torch
from torch import nn
from models.base_model import Net
from models.model_1 import BIG, BCM
from models.module.common import conv3x3_bn_pRelu, conv3x3_bn_pRelu
from models.backbone.PVT_V2 import Attention
import torch.nn.functional as F
from models.module.attention import SpatialAttention, ChannelAttention, SegNext_Attention


# feature Fusion module
class FFM(nn.Module):
    def __init__(self, in_left, in_down, num_heads=8, sr_ratio=8):  # left down
        super(FFM, self).__init__()
        self.up_conv = conv3x3_bn_pRelu(in_down, in_left, 1, 1)
        self.ca_att = ChannelAttention(in_left)
        self.sa_att = SpatialAttention()
        self.conv_fuse = conv3x3_bn_pRelu(in_left * 2, in_left, 1, 1)
        self.conv_left = conv3x3_bn_pRelu(in_left * 2, in_left, 1, 1)
        self.conv_down = conv3x3_bn_pRelu(in_left * 2, in_left, 1, 1)
        self.conv_final = conv3x3_bn_pRelu(in_left * 2, in_left, 1, 1)
        # self.att = SegNext_Attention(in_left)
        self.att = Attention(dim=in_left,
                             num_heads=num_heads,
                             qkv_bias=True,
                             sr_ratio=sr_ratio)

    def forward(self, left, down):
        B, _, H, W = left.shape
        if down.shape[2:] != left.shape[2:]:
            down = self.up_conv(F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=False))
        sa = self.sa_att(down)
        ca = self.ca_att(left)
        fuse = self.conv_fuse(torch.cat([sa * left, ca * down], 1))
        fuse = self.att(fuse.reshape(B, H * W, _), H, W).reshape(B, _, H, W)
        # fuse = self.att(fuse)
        left = self.conv_left(torch.cat([fuse, left], 1))
        down = self.conv_down(torch.cat([fuse, down], 1))
        return self.conv_final(torch.cat([left, down], 1))


class Decoder(nn.Module):
    def __init__(self, in_channel_list, img_size=224, channel=64):
        super(Decoder, self).__init__()
        set_channels = channel
        self.down1 = conv3x3_bn_pRelu(in_channel_list[0], set_channels)
        self.down2 = conv3x3_bn_pRelu(in_channel_list[1], set_channels)
        self.down3 = conv3x3_bn_pRelu(in_channel_list[2], set_channels)
        self.down4 = conv3x3_bn_pRelu(in_channel_list[3], set_channels)

        self.cff1 = FFM(set_channels, set_channels, sr_ratio=8)
        self.cff2 = FFM(set_channels, set_channels, sr_ratio=4)
        self.cff3 = FFM(set_channels, set_channels, sr_ratio=2)

        self.cff4 = FFM(set_channels, set_channels, sr_ratio=8)
        self.cff5 = FFM(set_channels, set_channels, sr_ratio=4)

        self.cff6 = FFM(set_channels, set_channels, sr_ratio=8)

        self.big = BIG(set_channels)
        self.bcm = BCM(set_channels)
        self.conv_bcm = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_sal = nn.Conv2d(set_channels, 1, 1, 1)

    def forward(self, fea, shape):
        fea1 = self.down1(fea[0])
        fea2 = self.down2(fea[1])
        fea3 = self.down3(fea[2])
        fea4 = self.down4(fea[3])

        bcm = self.bcm(fea1, fea2, fea3, fea4)  # [bs, 64, h, w]

        out3_1 = self.cff3(fea3, fea4)
        out2_1 = self.cff2(fea2, out3_1)
        out1_1 = self.cff1(fea1, out2_1)

        out2_2 = self.cff5(out2_1, out3_1)
        out1_2 = self.cff4(out1_1, out2_2)

        out1_3 = self.cff6(out1_2, out2_2)

        bcm = self.conv_bcm(bcm)
        sal = self.conv_sal(self.big(out1_3, bcm))
        bcm = F.interpolate(bcm, size=shape, mode='bilinear', align_corners=True)
        sal = F.interpolate(sal, size=shape, mode='bilinear', align_corners=True)

        return sal, bcm


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

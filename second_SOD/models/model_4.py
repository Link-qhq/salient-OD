import torch
from torch import nn
from models.backbone.Encoder_pvt import Encoder
from models.base_model import Net
from models.model_1 import BIG, BCM
from models.module.common import conv3x3_bn_relu, conv1x1_bn_relu
from models.backbone.resnet import ResNet
import torch.nn.functional as F
from models.module.attention import SpatialAttention, ChannelAttention, SegNext_Attention


# 基于model_1的改进
# feature Fusion module
# Cross guide module
# 引导两个分支之间的相互交流，充分利用各层次上全局特征和细节特征的复杂语义信息
class MCGM(nn.Module):
    def __init__(self, channel):
        super(MCGM, self).__init__()
        self.up_conv = conv3x3_bn_relu(channel, channel)
        # dt
        self.conv1_1 = conv3x3_bn_relu(channel, channel)
        self.conv1_2 = conv3x3_bn_relu(channel, channel)
        # gb
        self.conv2_1 = conv3x3_bn_relu(channel, channel)
        self.conv2_2 = conv3x3_bn_relu(channel, channel)

        self.conv3_1 = conv3x3_bn_relu(channel * 2, channel)
        self.conv3_2 = conv3x3_bn_relu(channel * 2, channel)

        self.conv4_1 = conv3x3_bn_relu(channel, channel)
        self.conv4_2 = conv3x3_bn_relu(channel, channel)

    def forward(self, gb, dt):  # global and detail branches => left down
        if dt.shape != gb.shape:
            dt = self.up_conv(F.interpolate(dt, size=gb.shape[2:], mode='bilinear', align_corners=True))
        dt1, gb1 = self.conv1_1(dt), self.conv2_1(gb)
        dt2, gb2 = self.conv1_2(dt1), self.conv2_2(gb1)
        cro_feat = torch.cat((dt2, gb2), dim=1)
        dt3, gb3 = self.conv3_1(cro_feat), self.conv3_2(cro_feat)
        f1 = dt1 + dt3
        f2 = gb1 + gb3
        dt4 = self.conv4_1(f1)
        gb4 = self.conv4_2(f2)
        return gb4, dt4


class Decoder(nn.Module):
    def __init__(self, in_channel_list, img_size=224, channel=64):
        super(Decoder, self).__init__()
        set_channels = channel
        self.down1 = conv3x3_bn_relu(in_channel_list[0], set_channels)
        self.down2 = conv3x3_bn_relu(in_channel_list[1], set_channels)
        self.down3 = conv3x3_bn_relu(in_channel_list[2], set_channels)
        self.down4 = conv3x3_bn_relu(in_channel_list[3], set_channels)

        self.cff1 = MCGM(set_channels)
        self.cff2 = MCGM(set_channels)
        self.cff3 = MCGM(set_channels)

        self.cff4 = MCGM(set_channels)
        self.cff5 = MCGM(set_channels)

        self.cff6 = MCGM(set_channels)

        self.big = BIG(set_channels)
        self.big1 = BIG(set_channels)
        self.big2 = BIG(set_channels)
        self.big3 = BIG(set_channels)
        self.bcm = BCM(set_channels)
        self.conv_bcm = nn.Conv2d(set_channels, 1, 3, 1, 1)
        self.conv_sal = nn.Conv2d(set_channels, 1, 3, 1, 1)
        self.conv_sal1 = nn.Conv2d(set_channels, 1, 3, 1, 1)
        self.conv_sal2 = nn.Conv2d(set_channels, 1, 3, 1, 1)
        self.conv_sal3 = nn.Conv2d(set_channels, 1, 3, 1, 1)

    def forward(self, fea, shape):
        fea1 = self.down1(fea[0])
        fea2 = self.down2(fea[1])
        fea3 = self.down3(fea[2])
        fea4 = self.down4(fea[3])

        bcm = self.bcm(fea1, fea2, fea3, fea4)  # [bs, 64, h, w]

        up3_1, right3_1 = self.cff3(fea3, fea4)
        up2_1, right2_1 = self.cff2(fea2, up3_1)
        up1_1, right1_1 = self.cff1(fea1, up2_1)

        up2_2, right2_2 = self.cff5(right2_1, right3_1)
        up1_2, right1_2 = self.cff4(right1_1, up2_2)

        up1_3, right1_3 = self.cff6(right1_2, right2_2)

        bcm = self.conv_bcm(bcm)
        sal = self.conv_sal(self.big(right1_3, bcm))
        sal3 = self.conv_sal3(self.big3(up1_3, bcm))
        sal2 = self.conv_sal2(self.big2(up1_2, bcm))
        sal1 = self.conv_sal1(self.big1(up1_1, bcm))
        bcm = F.interpolate(bcm, size=shape, mode='bilinear', align_corners=True)
        sal = F.interpolate(sal, size=shape, mode='bilinear', align_corners=True)
        sal3 = F.interpolate(sal3, size=shape, mode='bilinear', align_corners=True)
        sal2 = F.interpolate(sal2, size=shape, mode='bilinear', align_corners=True)
        sal1 = F.interpolate(sal1, size=shape, mode='bilinear', align_corners=True)

        return sal, bcm, sal3, sal2, sal1


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

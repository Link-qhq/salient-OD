import torch
from torch import nn
from models.backbone.Encoder_pvt import Encoder
from models.module.common import conv3x3_bn_relu, conv1x1_bn_relu
from models.backbone.resnet import ResNet
import torch.nn.functional as F
from models.module.attention import SpatialAttention, ChannelAttention, SegNext_Attention


class BIG(nn.Module):
    """
        boundary information guidance 边界信息引导模块
    """

    def __init__(self, channel, groups=16):
        super(BIG, self).__init__()
        self.gate1 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.ReLU(True),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   )
        self.gate2 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.ReLU(True),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   )
        self.gate3 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.ReLU(True),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   )
        self.gate4 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.ReLU(True),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   )

        self.self_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, channel, 1),
            nn.ReLU(True),
            nn.Conv2d(channel, 4, 1),
            nn.Softmax(1)
        )
        self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=1, padding=1),
                                  nn.GroupNorm(groups, channel), nn.ReLU(True))

        self.channel = channel
        self.weight = nn.Sequential(
            # nn.Conv2d(4, 8, 1, 1),
            # nn.ReLU(True),
            # nn.Conv2d(8, 4, 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x, edge):
        """
        :param x: torch.Size([1, channel(↑), 56, 56])
        :param edge: torch.Size([1, 1, 56, 56])
        :return:
        """
        x1, x2, x3, x4 = torch.split(x, self.channel // 4, dim=1)  # 16

        cm1 = self.gate1(x1)
        cm2 = self.gate2(x2)
        cm3 = self.gate3(x3)
        cm4 = self.gate4(x4)

        e1 = cm1 * torch.sigmoid(edge)
        e2 = cm2 * torch.sigmoid(edge)
        e3 = cm3 * torch.sigmoid(edge)
        e4 = cm4 * torch.sigmoid(edge)

        w1, w2, w3, w4 = torch.split(self.self_attention(torch.cat([e1, e2, e3, e4], 1)), 1, dim=1)

        nx1 = x1 * w1
        nx2 = x2 * w2
        nx3 = x3 * w3
        nx4 = x4 * w4

        return self.conv(torch.cat((nx1, nx2, nx3, nx4), 1))


# feature Fusion module
class FFM(nn.Module):
    def __init__(self, in_left, in_down):  # left down
        super(FFM, self).__init__()
        self.up_conv = conv3x3_bn_relu(in_down, in_left, 1, 1)
        self.ca_att = ChannelAttention(in_left)
        self.sa_att = SpatialAttention()
        self.conv_fuse = conv3x3_bn_relu(in_left * 2, in_left, 1, 1)
        self.conv_left = conv3x3_bn_relu(in_left * 2, in_left, 1, 1)
        self.conv_down = conv3x3_bn_relu(in_left * 2, in_left, 1, 1)
        self.conv_final = conv3x3_bn_relu(in_left * 2, in_left, 1, 1)
        self.att = SegNext_Attention(in_left)

    def forward(self, left, down):
        B, _, H, W = left.shape
        if down.shape[2:] != left.shape[2:]:
            down = self.up_conv(F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=False))
        sa = self.sa_att(down)
        ca = self.ca_att(left)
        fuse = self.conv_fuse(torch.cat([sa * left, ca * down], 1))
        # fuse = self.att(fuse.reshape(B, H * W, _)).reshape(B, _, H, W)
        fuse = self.att(fuse)
        left = self.conv_left(torch.cat([fuse, left], 1))
        down = self.conv_down(torch.cat([fuse, down], 1))
        return self.conv_final(torch.cat([left, down], 1))


# boundary
class BCM(nn.Module):
    def __init__(self, in_ch):
        super(BCM, self).__init__()

        self.conv1 = conv3x3_bn_relu(in_ch, in_ch)
        self.conv2 = conv3x3_bn_relu(in_ch, in_ch)
        self.conv3 = conv3x3_bn_relu(in_ch, in_ch)
        self.conv4 = conv3x3_bn_relu(in_ch, in_ch)

    def forward(self, f1, f2, f3, f4):  # 64 64 64 64
        f4 = self.conv4(F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=True))
        f3 = self.conv3(F.interpolate(f4 + f3, size=f2.shape[2:], mode='bilinear', align_corners=True))
        f2 = self.conv2(F.interpolate(f3 + f2, size=f1.shape[2:], mode='bilinear', align_corners=True))
        f1 = self.conv1(f2 + f1)
        return f1


class Decoder(nn.Module):
    def __init__(self, in_channel_list, img_size=224, channel=64):
        super(Decoder, self).__init__()
        set_channels = channel
        self.down1 = conv3x3_bn_relu(in_channel_list[0], set_channels)
        self.down2 = conv3x3_bn_relu(in_channel_list[1], set_channels)
        self.down3 = conv3x3_bn_relu(in_channel_list[2], set_channels)
        self.down4 = conv3x3_bn_relu(in_channel_list[3], set_channels)

        self.cff1 = FFM(set_channels, set_channels)
        self.cff2 = FFM(set_channels, set_channels)
        self.cff3 = FFM(set_channels, set_channels)

        self.cff4 = FFM(set_channels, set_channels)
        self.cff5 = FFM(set_channels, set_channels)

        self.cff6 = FFM(set_channels, set_channels)

        self.big = BIG(set_channels)
        self.bcm = BCM(set_channels)
        self.conv_bcm = nn.Conv2d(set_channels, 1, 3, 1, 1)
        self.conv_sal = nn.Conv2d(set_channels, 1, 3, 1, 1)

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


class Net(nn.Module):
    """
        Detail-guided salient object detection network
        Progressively guided network of detailed information
    """

    def __init__(self, backbone='resnet', img_size=224, channel=64, salient_idx=0):
        super(Net, self).__init__()
        self.img_size = [img_size, img_size]
        self.salient_idx = salient_idx
        self.backbone = backbone
        self.encoder = Encoder() if backbone == 'pvt' else ResNet()  # pvt or resnet or vgg
        self.in_channel_list = [64, 128, 320, 512] if backbone == 'pvt' else [256, 512, 1024, 2048]
        self.decoder = Decoder(in_channel_list=self.in_channel_list, channel=channel)

    def forward(self, x, shape=None):
        B, C, H, W = x.shape
        fea1, fea2, fea3, fea4 = self.encoder(x)
        if self.backbone == 'pvt':
            fea1 = fea1.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[0], H // 4, W // 4)
            fea2 = fea2.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[1], H // 8, W // 8)
            fea3 = fea3.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[2], H // 16, W // 16)
            fea4 = fea4.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[3], H // 32, W // 32)

        out = self.decoder((fea1, fea2, fea3, fea4), x.shape[2:] if shape is None else shape)
        return out

    def cal_mae(self, pred, gt):
        total_mae, avg_mae, B = 0.0, 0.0, pred.shape[0]
        pred = torch.sigmoid(pred)
        with torch.no_grad():
            for b in range(B):
                total_mae += torch.abs(pred[b][0] - gt[b]).mean()
        return total_mae / B

if __name__ == '__main__':
    from thop import profile

    input = torch.rand(2, 3, 224, 224)
    edge = torch.rand(2, 1, 56, 56)
    model = Net(backbone='pvt')
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

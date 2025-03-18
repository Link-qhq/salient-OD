import torch
from torch import nn
from models.module.attention import SpatialAttention
from models.module.common import conv3x3_bn_relu
from models.backbone.resnet import ResNet
import torch.nn.functional as F


class CFRM(nn.Module):
    def __init__(self, channel=64, kernel_size=5):  # 适用64
        super(CFRM, self).__init__()
        self.conv_left_2 = conv3x3_bn_relu(channel, channel)
        self.conv_down_2 = conv3x3_bn_relu(channel, channel)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self._x = nn.Conv2d(channel * 2, channel, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, kernel_size // 2))
        self._y = nn.Conv2d(channel * 2, channel, kernel_size=(kernel_size, 1), stride=(1, 1), padding=(kernel_size // 2, 0))
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.conv_sigmoid = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
            nn.Sigmoid()
        )

    def forward(self, left, down):
        if down.shape[2:] != left.shape[2:]:
            down = F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=True)
        left_2 = self.conv_left_2(left)
        down_2 = self.conv_down_2(down)
        fuse = self.conv_fuse(torch.cat([left_2, down_2], 1))
        tmp = torch.cat([self.max(fuse), self.avg(fuse)], 1)
        tmp = self.conv_sigmoid(torch.cat([self._x(tmp), self._y(tmp)], 1))
        up = left_2 * tmp
        right = down_2 * tmp
        return up, right


class AddFuse(nn.Module):
    def __init__(self, channel=64):
        super(AddFuse, self).__init__()
        # self.conv = conv3x3_bn_relu(channel, channel, 1, 1)
        self.eps = 1e-5

    def forward(self, up, fea):
        if up.shape[2:] != fea.shape[2:]:
            up = F.interpolate(up, size=fea.shape[2:], mode='bilinear', align_corners=True)
        # 空间注意力
        # context = (up.pow(2).sum(1, keepdims=True) + self.eps).pow(0.5)  # [B, 1, H, W] L2 正则化
        return up * fea


class SubEncoder(nn.Module):
    def __init__(self):
        super(SubEncoder, self).__init__()
        self.cfr1_1 = CFRM()
        self.cfr2_1 = CFRM()
        self.cfr3_1 = CFRM()
        self.af1_1 = AddFuse()
        self.af2_1 = AddFuse()
        self.af3_1 = AddFuse()
        self.af4_1 = AddFuse()

        self.cfr1_2 = CFRM()
        self.cfr2_2 = CFRM()
        self.cfr3_2 = CFRM()
        self.af1_2 = AddFuse()
        self.af2_2 = AddFuse()
        self.af3_2 = AddFuse()
        self.af4_2 = AddFuse()

        self.ca_1 = SpatialAttention()
        self.ca_2 = SpatialAttention()
        self.conv_1 = nn.Conv2d(64, 1, 1, 1)
        self.conv_2 = nn.Conv2d(64, 1, 1, 1)

    def forward(self, fea1, fea2, fea3, fea4):

        up3_1, right3_1 = self.cfr3_1(fea3, fea4)
        up2_1, right2_1 = self.cfr2_1(fea2, up3_1)
        up1_1, right1_1 = self.cfr1_1(fea1, up2_1)

        # context1 = self.ca_1(up1_1)
        context1 = self.conv_1(up1_1)
        right1_1 = self.af1_1(context1, right1_1)
        right2_1 = self.af2_1(context1, right2_1)
        right3_1 = self.af3_1(context1, right3_1)
        right4_1 = self.af4_1(context1, fea4)

        up3_2, right3_2 = self.cfr3_2(right3_1, right4_1)
        up2_2, right2_2 = self.cfr2_2(right2_1, up3_2)
        up1_2, right1_2 = self.cfr1_2(right1_1, up2_2)

        # context2 = self.ca_2(up1_2)
        context2 = self.conv_2(up1_2)
        right1_2 = self.af1_2(context2, right1_2)
        right2_2 = self.af2_2(context2, right2_2)
        right3_2 = self.af3_2(context2, right3_2)
        right4_2 = self.af4_2(context2, right4_1)

        return context2, context1, right1_2, right2_2, right3_2, right4_2


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        set_channels = 64

        self.dfe1 = conv3x3_bn_relu(256, set_channels)  # 64
        self.dfe2 = conv3x3_bn_relu(512, set_channels)  # 64
        self.dfe3 = conv3x3_bn_relu(1024, set_channels)  # 64
        self.dfe4 = conv3x3_bn_relu(2048, set_channels)  # 64

        self.salient_sub = SubEncoder()
        self.edge_sub = SubEncoder()

        self.conv_fuse_0 = nn.Conv2d(2, 1, 1, 1)
        self.conv_fuse_1 = nn.Conv2d(2, 1, 1, 1)

        for i in range(4):
            setattr(self, 'conv_salient_{}'.format(i), nn.Conv2d(set_channels, 1, 1, 1))
            setattr(self, 'conv_edge_{}'.format(i), nn.Conv2d(set_channels, 1, 1, 1))

    def forward(self, fea, shape):
        fea1, fea2, fea3, fea4 = fea

        fea1 = self.dfe1(fea1)
        fea2 = self.dfe2(fea2)
        fea3 = self.dfe3(fea3)
        fea4 = self.dfe4(fea4)

        salient_out = self.salient_sub(fea1, fea2, fea3, fea4)
        edge_out = self.edge_sub(fea1, fea2, fea3, fea4)

        # out0 = self.conv_fuse_0(torch.cat([salient_out[0], edge_out[0]], 1))
        out1 = self.conv_fuse_1(torch.cat([salient_out[1], edge_out[1]], 1))
        out1 = F.interpolate(out1, size=shape, mode='bilinear', align_corners=True)

        salient = [F.interpolate(getattr(self, 'conv_salient_{}'.format(i - 2))(salient_out[i]) if i > 1 else salient_out[i],
                                 size=shape, mode='bilinear', align_corners=True)
                   for i in range(6)]
        edge = [
            F.interpolate(getattr(self, 'conv_edge_{}'.format(i - 2))(edge_out[i]) if i > 1 else edge_out[i],
                          size=shape, mode='bilinear',
                          align_corners=True)
            for i in range(6)]
        return out1, salient, edge
        #
        # up1 = F.interpolate(self.conv_up1(up1_1), size=shape, mode='bilinear', align_corners=True)
        # up2 = F.interpolate(self.conv_up2(up1_2), size=shape, mode='bilinear', align_corners=True)
        # right1 = F.interpolate(self.conv_right1(right1_2), size=shape, mode='bilinear', align_corners=True)
        # right2 = F.interpolate(self.conv_right2(right2_2), size=shape, mode='bilinear', align_corners=True)
        # right3 = F.interpolate(self.conv_right3(right3_2), size=shape, mode='bilinear', align_corners=True)
        # right4 = F.interpolate(self.conv_right4(right4_2), size=shape, mode='bilinear', align_corners=True)
        # return up1, up2, right1, right2, right3, right4


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
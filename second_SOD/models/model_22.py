import torch
from torch import nn
from models.backbone.resnet import ResNet
import torch.nn.functional as F
from models.modify import MSFAM


class CFRM(nn.Module):
    def __init__(self, channel=64, kernel_size=5, ratio=4):  # 适用64
        super(CFRM, self).__init__()
        self._x = nn.Conv2d(2, 1, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, kernel_size // 2))
        self._y = nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), stride=(1, 1), padding=(kernel_size // 2, 0))
        self.conv_out = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        inter_channels = channel // ratio
        # 局部注意力
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel)
        )
        # 全局注意力
        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel)
        )
        # 局部注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel)
        )
        # 全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Conv2d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel)
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
        )
        self.out = nn.Conv2d(channel, 1, 1, 1)

    def forward(self, left, down):
        if down.shape[2:] != left.shape[2:]:
            down = F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=True)

        left = left * torch.sigmoid(self.local_att1(left) + self.global_att1(left))
        down = down * torch.sigmoid(self.local_att2(down) + self.global_att2(down))
        fuse = self.conv_fuse(torch.cat([left, down], 1))
        avg_out = torch.mean(fuse, dim=1, keepdim=True)
        max_out, _ = torch.max(fuse, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], 1)
        out = fuse * torch.sigmoid(self.conv_out(torch.cat([self._x(x), self._y(x)], 1)))
        return out, self.out(out)


class SubEncoder(nn.Module):
    def __init__(self):
        super(SubEncoder, self).__init__()

        self.cfr1 = CFRM()
        self.cfr2 = CFRM()
        self.cfr3 = CFRM()
        self.conv = nn.Conv2d(64, 1, 1, 1)

    def forward(self, fea1, fea2, fea3, fea4):
        up3, right3 = self.cfr3(fea3, fea4)
        up2, right2 = self.cfr2(fea2, up3)
        up1, right1 = self.cfr1(fea1, up2)

        return right1, right2, right3, self.conv(fea4)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        set_channels = 64
        self.msfam = MSFAM()
        self.sub = SubEncoder()
        self.conv = nn.Conv2d(set_channels, 1, 1, 1)

    def forward(self, fea, shape):
        fea1, fea2, fea3, fea4 = fea

        fea1, fea2, fea3, fea4 = self.msfam(fea1, fea2, fea3, fea4)

        right1, right2, right3, right4 = self.sub(fea1, fea2, fea3, fea4)

        right1 = F.interpolate(right1, size=shape, mode='bilinear', align_corners=True)
        right2 = F.interpolate(right2, size=shape, mode='bilinear', align_corners=True)
        right3 = F.interpolate(right3, size=shape, mode='bilinear', align_corners=True)
        right4 = F.interpolate(right4, size=shape, mode='bilinear', align_corners=True)
        return right1, right2, right3, right4


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

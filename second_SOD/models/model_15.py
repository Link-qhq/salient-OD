import torch
from torch import nn
from models.methods.DSRNet import CCM
from models.module.SPP import SPPF_LSKA
from models.module.attention import ChannelAttention, SpatialAttention
from models.backbone.resnet import ResNet
import torch.nn.functional as F


class FFM(nn.Module):
    def __init__(self, channel):  # left down
        super(FFM, self).__init__()
        self.ca_att = ChannelAttention(channel)
        self.sa_att = SpatialAttention()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.conv_left = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.conv_down = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )

    def forward(self, left, down):
        if down.shape[2:] != left.shape[2:]:
            down = F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=True)
        sa = self.sa_att(left)
        ca = self.ca_att(down)
        down = down * sa
        left = left * ca
        fuse = self.conv_fuse(torch.cat([left, down], 1))
        left = self.conv_left(torch.cat([fuse, left], 1))
        down = self.conv_down(torch.cat([fuse, down], 1))
        final = self.conv_final(torch.cat([left, down], 1))
        return final


class Fuse(nn.Module):
    def __init__(self, channel=64):
        super(Fuse, self).__init__()
        self.conv_flow1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.conv_flow2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.conv_fea1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.conv_fea2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.maxpool_flow = nn.MaxPool2d(7, 1, 3)
        self.avgpool_flow = nn.AvgPool2d(7, 1, 3)
        self.conv_fuse_flow = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 7, 1, 3),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.maxpool_fea = nn.MaxPool2d(7, 1, 3)
        self.avgpool_fea = nn.AvgPool2d(7, 1, 3)
        self.conv_fuse_fea = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 7, 1, 3),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True)
        )

    def forward(self, flow, fea):
        if flow.shape[2:] != fea.shape[2:]:
            flow = F.interpolate(flow, size=fea.shape[2:], mode='bilinear', align_corners=True)
        flow1 = self.conv_flow1(flow)
        flow2 = self.conv_flow2(flow1)

        fea1 = self.conv_fea1(fea)
        fea2 = self.conv_fea2(fea1)

        fuse_flow = flow1 * fea2
        fuse_fea = fea1 * flow2
        flow_cat = self.conv_fuse_flow(
            torch.cat([self.maxpool_flow(fuse_flow), self.avgpool_flow(fuse_flow)], 1)
        )
        fea_cat = self.conv_fuse_fea(
            torch.cat([self.maxpool_fea(fuse_fea), self.avgpool_fea(fuse_fea)], 1)
        )
        out = self.conv_final(flow_cat + fea_cat)
        return out


class Flow(nn.Module):
    def __init__(self):
        super(Flow, self).__init__()
        self.fuse4 = Fuse()
        self.fuse3 = Fuse()
        self.fuse2 = Fuse()
        self.fuse1 = Fuse()

    def forward(self, flow, spp):
        spp4 = self.fuse4(flow, spp)
        spp4 = F.interpolate(spp4, scale_factor=2, mode='bilinear', align_corners=True)
        spp3 = self.fuse3(flow, spp4)
        spp3 = F.interpolate(spp3, scale_factor=2, mode='bilinear', align_corners=True)
        spp2 = self.fuse2(flow, spp3)
        spp2 = F.interpolate(spp2, scale_factor=2, mode='bilinear', align_corners=True)
        spp1 = self.fuse1(flow, spp2)
        return spp1, spp2, spp3, spp4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        set_channels = 64

        self.dfe1 = CCM(256, set_channels, 4)  # 64
        self.dfe2 = CCM(512, set_channels, 4)  # 64
        self.dfe3 = CCM(1024, set_channels, 4)  # 64
        self.dfe4 = CCM(2048, set_channels, 4)  # 64

        self.ffm1 = FFM(set_channels)
        self.ffm2 = FFM(set_channels)
        self.ffm3 = FFM(set_channels)
        self.ffm4 = FFM(set_channels)

        self.spp = SPPF_LSKA(2048, set_channels, 5)

        self.flow = Flow()
        self.conv_spp1 = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_spp2 = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_spp3 = nn.Conv2d(set_channels, 1, 1, 1)
        self.conv_spp4 = nn.Conv2d(set_channels, 1, 1, 1)

    def forward(self, fea, shape):
        fea1, fea2, fea3, fea4 = fea

        spp = self.spp(fea4)

        fea1 = self.dfe1(fea1)
        fea2 = self.dfe2(fea2)
        fea3 = self.dfe3(fea3)
        fea4 = self.dfe4(fea4)

        out4_1 = self.ffm4(fea4, spp)
        out3_1 = self.ffm3(fea3, out4_1)
        out2_1 = self.ffm2(fea2, out3_1)
        out1_1 = self.ffm1(fea1, out2_1)

        spp1, spp2, spp3, spp4 = self.flow(out1_1, spp)
        spp1 = F.interpolate(self.conv_spp1(spp1), size=shape, mode='bilinear', align_corners=True)
        spp2 = F.interpolate(self.conv_spp2(spp2), size=shape, mode='bilinear', align_corners=True)
        spp3 = F.interpolate(self.conv_spp3(spp3), size=shape, mode='bilinear', align_corners=True)
        spp4 = F.interpolate(self.conv_spp4(spp4), size=shape, mode='bilinear', align_corners=True)
        return spp1, spp2, spp3, spp4


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
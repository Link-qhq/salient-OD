import math

import torch
from torch import nn
from models.backbone.resnet import ResNet
import torch.nn.functional as F
from models.methods.GPONet import GFN, CGM, FFM
from models.backbone.Encoder_pvt import Encoder


class Decoder(nn.Module):
    def __init__(self, in_channel_list=None):
        super(Decoder, self).__init__()
        if in_channel_list is None:
            self.in_channel_list = [256, 512, 1024, 2048]
        else:
            self.in_channel_list = in_channel_list
        set_channels = 64
        self.channel = set_channels

        self.gfn1 = GFN(channel_list=in_channel_list, channel=set_channels)
        self.gfn2 = GFN(channel_list=in_channel_list, channel=set_channels)

        self.cgm1 = CGM(channel=set_channels)
        self.cgm2 = CGM(channel=set_channels)
        self.cgm3 = CGM(channel=set_channels)
        self.cgm4 = CGM(channel=set_channels)

        self.gb_head1 = nn.Conv2d(set_channels, 1, 3, padding=1)
        self.gb_head2 = nn.Conv2d(set_channels, 1, 3, padding=1)
        self.gb_head3 = nn.Conv2d(set_channels, 1, 3, padding=1)
        self.gb_head4 = nn.Conv2d(set_channels, 1, 3, padding=1)

        self.eg_head1 = nn.Conv2d(set_channels, 1, 3, padding=1)
        self.eg_head2 = nn.Conv2d(set_channels, 1, 3, padding=1)
        self.eg_head3 = nn.Conv2d(set_channels, 1, 3, padding=1)
        self.eg_head4 = nn.Conv2d(set_channels, 1, 3, padding=1)

        self.ffm = FFM(fuse_channel=set_channels)
        self.fuse_head = nn.Conv2d(8, 1, 3, 1, 1)

    def forward(self, fea, shape):
        if self.in_channel_list[0] == 64:
            B, _, C = fea[0].shape
            # torch.permute()
            fea[0] = fea[0].reshape(B, int(math.sqrt(_)), int(math.sqrt(_)), self.in_channel_list[0]).permute(0, 3, 1, 2)
            fea[1] = fea[1].reshape(B, int(math.sqrt(_)) // 2, int(math.sqrt(_)) // 2, self.in_channel_list[1]).permute(0, 3, 1, 2)
            fea[2] = fea[2].reshape(B, int(math.sqrt(_)) // 4, int(math.sqrt(_)) // 4, self.in_channel_list[2]).permute(0, 3, 1, 2)
            fea[3] = fea[3].reshape(B, int(math.sqrt(_)) // 8, int(math.sqrt(_)) // 8, self.in_channel_list[3]).permute(0, 3, 1, 2)
        sal_fea = self.gfn1(fea)
        edge_fea = self.gfn2(fea)

        out4_sal, out4_edge = self.cgm4(sal_fea[3], edge_fea[3])
        out3_sal, out3_edge = self.cgm3(sal_fea[2], edge_fea[2])
        out2_sal, out2_edge = self.cgm2(sal_fea[1], edge_fea[1])
        out1_sal, out1_edge = self.cgm1(sal_fea[0], edge_fea[0])

        out4_sal = F.interpolate(self.gb_head4(out4_sal), size=shape, mode='bilinear', align_corners=True)
        out3_sal = F.interpolate(self.gb_head3(out3_sal), size=shape, mode='bilinear', align_corners=True)
        out2_sal = F.interpolate(self.gb_head2(out2_sal), size=shape, mode='bilinear', align_corners=True)
        out1_sal = F.interpolate(self.gb_head1(out1_sal), size=shape, mode='bilinear', align_corners=True)

        out4_edge = F.interpolate(self.eg_head4(out4_edge), size=shape, mode='bilinear', align_corners=True)
        out3_edge = F.interpolate(self.eg_head3(out3_edge), size=shape, mode='bilinear', align_corners=True)
        out2_edge = F.interpolate(self.eg_head2(out2_edge), size=shape, mode='bilinear', align_corners=True)
        out1_edge = F.interpolate(self.eg_head1(out1_edge), size=shape, mode='bilinear', align_corners=True)

        cat_pred = torch.cat([out4_sal, out3_sal, out2_sal, out1_sal, out4_edge, out3_edge, out2_edge, out1_edge], 1)
        cat_pred = self.ffm(cat_pred)
        cat_pred = F.interpolate(self.fuse_head(cat_pred), size=shape, mode='bilinear', align_corners=True)
        return [out4_sal, out3_sal, out2_sal, out1_sal], [out4_edge, out3_edge, out2_edge, out1_edge], cat_pred


class PGN(nn.Module):
    """
        Detail-guided salient object detection network
        Progressively guided network of detailed information
    """

    def __init__(self, backbone='resnet', img_size=224):
        super(PGN, self).__init__()
        self.img_size = [img_size, img_size]
        self.backbone = backbone
        self.encoder = Encoder() if backbone == 'pvt' else ResNet()
        self.in_channel_list = [64, 128, 320, 512] if backbone == 'pvt' else [256, 512, 1024, 2048]
        self.decoder = Decoder(in_channel_list=self.in_channel_list)

    def forward(self, x, shape=None):
        fea = self.encoder(x)
        out = self.decoder(fea, x.shape[2:] if shape is None else shape)
        return out


if __name__ == '__main__':
    from thop import profile

    input = torch.rand(2, 3, 352, 352)
    edge = torch.rand(2, 1, 56, 56)
    model = PGN()
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
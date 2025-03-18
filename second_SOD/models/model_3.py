import math
import torch
from torch import nn
import torch.nn.functional as F
from models.methods.GPONet import GFN, CGM, FFM


class Decoder(nn.Module):
    def __init__(self, in_channel_list=None, img_size=224, channel=64):
        super(Decoder, self).__init__()
        if in_channel_list is None:
            self.in_channel_list = [256, 512, 1024, 2048]
        else:
            self.in_channel_list = in_channel_list
        set_channels = channel
        self.channel = set_channels

        self.gfn1 = GFN(channel_list=in_channel_list, channel=set_channels)
        self.gfn2 = GFN(channel_list=in_channel_list, channel=set_channels)

        self.cgm1 = CGM(channel=set_channels)
        self.cgm2 = CGM(channel=set_channels)
        self.cgm3 = CGM(channel=set_channels)
        self.cgm4 = CGM(channel=set_channels)

        self.gb_head1 = nn.Conv2d(set_channels, 1, 1)
        self.gb_head2 = nn.Conv2d(set_channels, 1, 1)
        self.gb_head3 = nn.Conv2d(set_channels, 1, 1)
        self.gb_head4 = nn.Conv2d(set_channels, 1, 1)

        self.eg_head1 = nn.Conv2d(set_channels, 1, 1)
        self.eg_head2 = nn.Conv2d(set_channels, 1, 1)
        self.eg_head3 = nn.Conv2d(set_channels, 1, 1)
        self.eg_head4 = nn.Conv2d(set_channels, 1, 1)

        self.ffm = FFM(fuse_channel=set_channels)
        self.fuse_head = nn.Conv2d(8, 1, 1, 1, 1)

    def forward(self, fea, shape):
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
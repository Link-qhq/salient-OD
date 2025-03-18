import torch
from torch import nn
from models.module.common import conv3x3_bn_relu
from models.backbone.resnet import ResNet
from models.methods.ICONet import asyConv
import torch.nn.functional as F
from models.module.attention import SpatialAttention, ChannelAttention, SegNext_Attention


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
        # self.att = Token_performer(dim=emb_dim, in_dim=emb_dim, kernel_ratio=0.5)
        # self.att = ResCBAM(channel)
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
        # return left, down


class BCM(nn.Module):
    def __init__(self, in_ch):
        super(BCM, self).__init__()
        # self.pre1 = nn.Sequential(
        #     nn.Conv2d(256, in_ch, 3, 1, 1),
        #     nn.BatchNorm2d(in_ch),
        #     nn.PReLU()
        # )
        # self.pre2 = nn.Sequential(
        #     nn.Conv2d(512, in_ch * 2, 3, 1, 1),
        #     nn.BatchNorm2d(in_ch * 2),
        #     nn.PReLU()
        # )
        # self.pre3 = nn.Sequential(
        #     nn.Conv2d(1024, in_ch * 4, 3, 1, 1),
        #     nn.BatchNorm2d(in_ch * 4),
        #     nn.PReLU()
        # )
        # self.pre4 = nn.Sequential(
        #     nn.Conv2d(2048, in_ch * 8, 3, 1, 1),
        #     nn.BatchNorm2d(in_ch * 8),
        #     nn.PReLU()
        # )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 3, 1, 1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch * 4, in_ch * 2, 3, 1, 1),
            nn.BatchNorm2d(in_ch * 2),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch * 8, in_ch * 4, 3, 1, 1),
            nn.BatchNorm2d(in_ch * 4),
            nn.PReLU()
        )

    def forward(self, f1, f2, f3, f4):  # 256 512 1024 2048
        # f1 = self.pre1(f1)
        # f2 = self.pre2(f2)
        # f3 = self.pre3(f3)
        # f4 = self.pre4(f4)

        f4 = self.conv4(F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=False))
        f3 = self.conv3(F.interpolate(f4 + f3, size=f2.shape[2:], mode='bilinear', align_corners=False))
        f2 = self.conv2(F.interpolate(f3 + f2, size=f1.shape[2:], mode='bilinear', align_corners=False))
        f1 = self.conv1(f2 + f1)
        return f1


class DFEM(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super(DFEM, self).__init__()
        self.oriConv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
        self.asyConv = nn.Sequential(
            asyConv(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                    padding_mode='zeros', deploy=False)
        )
        self.atrConv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, dilation=dilation, padding=dilation, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(torch.cat([self.asyConv(x), self.oriConv(x), self.atrConv(x)], 1))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        set_channels = 64

        self.dfe1 = DFEM(256, set_channels)  # 64
        self.dfe2 = DFEM(512, set_channels * 2)  # 128
        self.dfe3 = DFEM(1024, set_channels * 4)  # 256
        self.dfe4 = DFEM(2048, set_channels * 8)  # 512

        self.cff1 = FFM(set_channels, set_channels * 2)
        self.cff2 = FFM(set_channels * 2, set_channels * 4)
        self.cff3 = FFM(set_channels * 4, set_channels * 8)

        self.cff4 = FFM(set_channels, set_channels * 2)
        self.cff5 = FFM(set_channels * 2, set_channels * 4)

        self.cff6 = FFM(set_channels, set_channels * 2)

        self.bcm = BCM(set_channels)
        self.sal = nn.Conv2d(set_channels, 1, 1, 1)
        self.edge = nn.Conv2d(set_channels, 1, 1, 1)

    def forward(self, fea, shape):
        fea1, fea2, fea3, fea4 = fea

        fea1 = self.dfe1(fea1)
        fea2 = self.dfe2(fea2)
        fea3 = self.dfe3(fea3)
        fea4 = self.dfe4(fea4)

        bcm = self.bcm(fea1, fea2, fea3, fea4)  # [bs, 64, h, w]

        out3_1 = self.cff3(fea3, fea4)
        out2_1 = self.cff2(fea2, out3_1)
        out1_1 = self.cff1(fea1, out2_1)

        out2_2 = self.cff5(out2_1, out3_1)
        out1_2 = self.cff4(out1_1, out2_2)

        out1_3 = self.cff6(out1_2, out2_2)

        out = F.interpolate(self.sal(out1_3 + bcm), size=shape, mode='bilinear', align_corners=False)
        bcm = F.interpolate(self.edge(bcm), size=shape, mode='bilinear', align_corners=False)
        return out, bcm


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

    input = torch.rand(2, 3, 224, 224)
    edge = torch.rand(2, 1, 56, 56)
    model = PGN()
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
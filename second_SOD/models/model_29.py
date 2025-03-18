import torch
from torch import nn
from models.methods.CSEPNet import BasicConv2d
from models.module.common import conv3x3_bn_relu
from models.backbone.resnet import ResNet
import torch.nn.functional as F
from models.module.attention import SpatialAttention, ChannelAttention, SegNext_Attention


class ReFine(nn.Module):
    def __init__(self, channel, mid_channel=32):
        super(ReFine, self).__init__()
        self.downsample = nn.Sequential()
        self.conv = nn.Sequential(nn.Conv2d(channel, mid_channel, kernel_size=(1, 1)),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_channel, mid_channel, kernel_size=(3, 3), padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_channel, 1, kernel_size=(3, 3), padding=1))
        # self.avg_pooling = nn.AvgPool2d(kernel_size=15, stride=1, padding=7)
        # * torch.abs(self.avg_pooling(attention) - attention)

    def forward(self, x, attention):
        x = x + attention
        x = self.conv(x)
        return x


def cus_sample(feat, **kwargs):
    """
    :param feat: 输入特征
    :param kwargs: size或者scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=True)


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
            nn.Conv2d(4, 16, 1),
            nn.ReLU(True),
            nn.Conv2d(16, 4, 1),
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
        # gv1 = F.avg_pool2d(e1, (e1.size(2), e1.size(3)), stride=(e1.size(2), e1.size(3)))
        # gv2 = F.avg_pool2d(e2, (e2.size(2), e2.size(3)), stride=(e2.size(2), e2.size(3)))
        # gv3 = F.avg_pool2d(e3, (e3.size(2), e3.size(3)), stride=(e3.size(2), e3.size(3)))
        # gv4 = F.avg_pool2d(e4, (e4.size(2), e4.size(3)), stride=(e4.size(2), e4.size(3)))
        # weight = self.weight(torch.cat((gv1, gv2, gv3, gv4), 1))
        # w1, w2, w3, w4 = torch.split(weight, 1, dim=1)

        nx1 = x1 * w1
        nx2 = x2 * w2
        nx3 = x3 * w3
        nx4 = x4 * w4

        return self.conv(torch.cat((nx1, nx2, nx3, nx4), 1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
        # self.att = Token_performer(dim=emb_dim, in_dim=emb_dim, kernel_ratio=0.5)
        # self.att = Attention(in_left)
        # self.att = ResCBAM(channel)
        self.att = SegNext_Attention(in_left)
        # self.att = SCSA(in_left, 8, gate_layer='sigmoid')
        # self.att = MixedAttentionBlock(dim=in_left, img_size=img_size, window_size=window_size, num_heads=num_heads,
        #                                mlp_ratio=mlp_ratio)

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


class CSC(nn.Module):
    def __init__(self, left_channel):
        super(CSC, self).__init__()
        self.upsample = cus_sample
        self.conv1 = BasicConv2d(left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.conv_cat = BasicConv2d(2 * left_channel, left_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fusion = BasicConv2d(left_channel, left_channel, 3, padding=1)

    def forward(self, left, right):
        right = self.upsample(right, scale_factor=2)  # right 上采样
        right = self.conv1(right)
        x1 = left.mul(self.sa2(right)) + left
        x2 = right.mul(self.sa1(left)) + right
        mid = self.conv_cat(torch.cat((x1, x2), 1))  # concat
        mid = self.conv2(mid)
        mid = self.sigmoid(mid)
        out = left * mid + right * mid
        out = self.fusion(out)

        return out


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
    def __init__(self, img_size=224):
        super(Decoder, self).__init__()
        set_channels = 64
        self.down1 = conv3x3_bn_relu(256, set_channels)
        self.down2 = conv3x3_bn_relu(512, set_channels)
        self.down3 = conv3x3_bn_relu(1024, set_channels)
        self.down4 = conv3x3_bn_relu(2048, set_channels)

        self.cff1 = FFM(set_channels, set_channels)
        self.cff2 = FFM(set_channels, set_channels)
        self.cff3 = FFM(set_channels, set_channels)

        self.cff4 = FFM(set_channels, set_channels)
        self.cff5 = FFM(set_channels, set_channels)

        self.cff6 = FFM(set_channels, set_channels)

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

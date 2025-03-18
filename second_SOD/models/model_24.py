import math
import torch
from torch import nn

from models.model_26 import DFEM
from models.module.common import conv3x3_bn_relu
from models.module.attention import ChannelAttention, SpatialAttention, SegNext_Attention
from torch.nn import functional as F
from models.methods.BIPGNet import BIG
from models.backbone.resnet import ResNet


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1):
        super().__init__()
        self.emb = in_dim * head_cnt  # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        y = self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = x + self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FFM(nn.Module):
    def __init__(self, channel, emb_dim):  # left down
        super(FFM, self).__init__()
        self.ca_att = ChannelAttention(channel)
        self.sa_att = SpatialAttention()
        self.conv_fuse = conv3x3_bn_relu(channel * 2, channel, 1, 1)
        self.conv_left = conv3x3_bn_relu(channel * 2, channel, 1, 1)
        self.conv_down = conv3x3_bn_relu(channel * 2, channel, 1, 1)
        self.conv_final = conv3x3_bn_relu(channel * 2, channel, 1, 1)
        # self.att = Token_performer(dim=emb_dim, in_dim=emb_dim, kernel_ratio=0.5)
        # self.att = ResCBAM(channel)
        self.att = SegNext_Attention(channel)

    def forward(self, left, down):
        B, _, H, W = left.shape
        if down.shape[2:] != left.shape[2:]:
            down = F.interpolate(down, size=left.shape[2:], mode='bilinear', align_corners=True)
        sa = self.sa_att(down)
        ca = self.ca_att(left)
        fuse = self.conv_fuse(torch.cat([sa * left, ca * down], 1))
        # fuse = self.att(fuse.reshape(B, H * W, _)).reshape(B, _, H, W)
        fuse = self.att(fuse)
        left = self.conv_left(torch.cat([fuse, left], 1))
        down = self.conv_down(torch.cat([fuse, down], 1))
        return left, down


class SubEncoder(nn.Module):
    def __init__(self, channel):
        super(SubEncoder, self).__init__()

        self.cfr1 = FFM(channel, channel)
        self.cfr2 = FFM(channel, channel)
        self.cfr3 = FFM(channel, channel)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, 1, 1, 1)
        )
        self.big1 = BIG(channel)
        self.big2 = BIG(channel)
        self.big3 = BIG(channel)
        self.big4 = BIG(channel)

    def forward(self, fea1, fea2, fea3, fea4):
        up3, right3 = self.cfr3(fea3, fea4)
        up2, right2 = self.cfr2(fea2, up3)
        up1, right1 = self.cfr1(fea1, up2)
        pred = self.conv(up1)
        right1 = self.big1(right1, F.interpolate(pred, size=right1.shape[2:], mode='bilinear', align_corners=False))
        right2 = self.big2(right2, F.interpolate(pred, size=right2.shape[2:], mode='bilinear', align_corners=False))
        right3 = self.big3(right3, F.interpolate(pred, size=right3.shape[2:], mode='bilinear', align_corners=False))
        right4 = self.big4(fea4, F.interpolate(pred, size=fea4.shape[2:], mode='bilinear', align_corners=False))

        return pred, right1, right2, right3, right4


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        set_channels = 64
        self.down4 = DFEM(2048, set_channels)
        self.down3 = DFEM(1024, set_channels)
        self.down2 = DFEM(512, set_channels)
        self.down1 = DFEM(256, set_channels)

        self.sub_1 = SubEncoder(set_channels)
        self.sub_2 = SubEncoder(set_channels)

        self.conv_right1 = nn.Sequential(
            nn.Conv2d(set_channels, set_channels, 3, 1, 1),
            nn.BatchNorm2d(set_channels),
            nn.Conv2d(set_channels, 1, 1, 1)
        )
        self.conv_right2 = nn.Sequential(
            nn.Conv2d(set_channels, set_channels, 3, 1, 1),
            nn.BatchNorm2d(set_channels),
            nn.Conv2d(set_channels, 1, 1, 1)
        )
        self.conv_right3 = nn.Sequential(
            nn.Conv2d(set_channels, set_channels, 3, 1, 1),
            nn.BatchNorm2d(set_channels),
            nn.Conv2d(set_channels, 1, 1, 1)
        )
        self.conv_right4 = nn.Sequential(
            nn.Conv2d(set_channels, set_channels, 3, 1, 1),
            nn.BatchNorm2d(set_channels),
            nn.Conv2d(set_channels, 1, 1, 1)
        )

    def forward(self, fea, shape):
        fea1 = self.down1(fea[0])
        fea2 = self.down2(fea[1])
        fea3 = self.down3(fea[2])
        fea4 = self.down4(fea[3])

        pred1, right1, right2, right3, right4 = self.sub_1(fea1, fea2, fea3, fea4)
        pred2, right1, right2, right3, right4 = self.sub_2(right1, right2, right3, right4)

        pred1 = F.interpolate(pred1, size=shape, mode='bilinear', align_corners=False)
        pred2 = F.interpolate(pred2, size=shape, mode='bilinear', align_corners=False)
        right1 = F.interpolate(self.conv_right1(right1), size=shape, mode='bilinear', align_corners=False)
        right2 = F.interpolate(self.conv_right2(right2), size=shape, mode='bilinear', align_corners=False)
        right3 = F.interpolate(self.conv_right3(right3), size=shape, mode='bilinear', align_corners=False)
        right4 = F.interpolate(self.conv_right4(right4), size=shape, mode='bilinear', align_corners=False)
        return pred1, pred2, right1, right2, right3, right4


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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.attention import ChannelAttention, SpatialAttention
from models.module.common import conv3x3_bn_relu
from models.module.Transformer import Transformer


class CCM(nn.Module):
    def __init__(self, infeature, out, redio):
        """ channel compression module (CCM) """
        super(CCM, self).__init__()
        self.down = nn.Conv2d(infeature, out, kernel_size=1, stride=1)
        self.channel_attention = ChannelAttention(out, redio)

    def forward(self, x):
        x = self.down(x)
        w = self.channel_attention(x)
        return x * w


class FusionEnhance(nn.Module):
    """
        特征融合增强模块
        fusion enhancement module (FEM)
    """
    def __init__(self, in_ch):
        super(FusionEnhance, self).__init__()
        self.in_ch = in_ch
        self.ca = ChannelAttention(self.in_ch, 4)
        self.sa = SpatialAttention()
        self.cat_conv = conv3x3_bn_relu(2 * in_ch, in_ch)
        self.in1_conv = conv3x3_bn_relu(2 * in_ch, in_ch)
        self.in2_conv = conv3x3_bn_relu(2 * in_ch, in_ch)
        self.out_cat = conv3x3_bn_relu(2 * in_ch, in_ch)
        self.fusion_conv = conv3x3_bn_relu(2 * in_ch, in_ch)

    def forward(self, in1, in2, in3=None):
        """
            in1 通道
            in2 空间
            in3 融合特征
        """
        in1_1 = self.ca(in1)
        in2_1 = self.sa(in2)
        in1_2 = in1 * in2_1
        in2_2 = in2 * in1_1
        in_cat = self.cat_conv(torch.cat([in1_2, in2_2], 1))
        in1_cat = self.in1_conv(torch.cat([in1, in_cat], 1))
        in2_cat = self.in2_conv(torch.cat([in2, in_cat], 1))
        out = self.out_cat(torch.cat([in1_cat, in2_cat], 1))
        if in3 is not None:
            return self.fusion_conv(torch.cat([out, in3], 1))
        return out


class FEM(nn.Module):
    def __init__(self, emb_dim=320, hw=7, cur_stg=512):
        super(FEM, self).__init__()

        self.shuffle = nn.PixelShuffle(2)
        self.unfold = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.concatFuse = nn.Sequential(nn.Linear(emb_dim + cur_stg // 4, emb_dim),
                                        nn.GELU(),
                                        nn.Linear(emb_dim, emb_dim))
        self.att = Token_performer(dim=emb_dim, in_dim=emb_dim, kernel_ratio=0.5)
        self.hw = hw
        self.fuse_enhance = fuse_enhance(cur_stg // 4)

    def forward(self, a, b, c):
        B, _, _ = b.shape
        # Transform shape and upsample
        b = self.shuffle(b.transpose(1, 2).reshape(B, -1, self.hw, self.hw))

        # Blend and then switch back
        b = self.fuse_enhance(b, c)
        b = self.unfold(b).transpose(1, 2)

        # cat then adjusts a full connection layer to the number of channels that transformer needs to input
        out = self.concatFuse(torch.cat([a, b], dim=2))
        out = self.att(out)

        return out


class fuse_enhance(nn.Module):
    def __init__(self, infeature):
        super(fuse_enhance, self).__init__()
        self.infeature = infeature
        # Channel attention
        self.ca = ChannelAttention(self.infeature)
        # Spatial attention
        self.sa = SpatialAttention()

        self.cbr1 = conv3x3_bn_relu(2 * self.infeature, self.infeature)
        self.cbr2 = conv3x3_bn_relu(2 * self.infeature, self.infeature)
        self.cbr3 = conv3x3_bn_relu(2 * self.infeature, self.infeature)
        self.cbr4 = conv3x3_bn_relu(2 * self.infeature, self.infeature)

        self.cbr5 = conv3x3_bn_relu(self.infeature, self.infeature)

    def forward(self, t, c):
        assert t.shape == c.shape, "cnn and transfrmer should have same size"
        # B, C, H, W = r.shape
        t_s = self.sa(t)  # Transformer space attention weight
        c_c = self.ca(c)  # CNN channel attention weight

        t_x = t * c_c
        c_x = c * t_s

        # Stepwise integration
        # 1
        x = torch.cat([t_x, c_x], dim=1)
        x = self.cbr1(x)
        # 2
        tx = torch.cat([t, x], dim=1)
        cx = torch.cat([c, x], dim=1)

        tx = self.cbr2(tx)
        cx = self.cbr3(cx)
        # 3
        x = torch.cat([tx, cx], dim=1)
        x = self.cbr4(x)

        out = self.cbr5(x)

        return out


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


class ERM(nn.Module):
    def __init__(self, inc, outc, hw, embed_dim, num_patches, depth=4):
        super(ERM, self).__init__()
        self.conv_p1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_p2 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_glb = nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True)

        self.conv_matt = nn.Sequential(nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(inc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))

        self.sigmoid = nn.Sigmoid()
        self.tf = Transformer(depth=depth,
                              num_heads=1,
                              embed_dim=embed_dim,
                              mlp_ratio=3,
                              num_patches=num_patches)
        self.hw = hw
        self.inc = inc

    def tensor_erode(self, bin_img, ksize=3):
        # padding is added to the original image first to prevent the image size from shrinking after corrosion
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # unfold the original image into a patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # Take the smallest value in each patch
        eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
        return eroded

    def tensor_dilate(self, bin_img, ksize=3):  #
        # padding is added to the original image first to prevent the image size from shrinking after corrosion
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
        # unfold the original image into a patch
        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)
        # B x C x H x W x k x k
        # Take the largest value in each patch
        dilate = patches.reshape(B, C, H, W, -1)
        dilate, _ = dilate.max(dim=-1)
        return dilate

    def forward(self, x):
        # x in shape of B,N,C
        B, _, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.hw, self.hw)

        x = self.conv_fuse(x)
        # pred1
        p1 = self.conv_p1(x)

        d = self.tensor_dilate(p1)
        e = self.tensor_erode(p1)
        matt = d - e
        matt = self.conv_matt(matt)
        fea = x * (1 + matt)  # refining

        # reshape x back to B,N,C
        fea = fea.reshape(B, self.inc, -1).transpose(1, 2)
        fea = self.tf(fea, True)  # Through transformer
        p2 = self.conv_p2(fea.transpose(1, 2).reshape(B, C, self.hw, self.hw)) # pred2

        return [p1, p2, fea]

if __name__ == '__main__':
    input1 = torch.rand(1, 256, 64, 64)
    input2 = torch.rand(1, 256, 64, 64)
    input3 = torch.rand(1, 256, 64, 64)
    net = FusionEnhance(256)
    print(net(input1, input2, input3).shape)
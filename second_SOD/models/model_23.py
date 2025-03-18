import math

import numpy as np
import torch
from torch import nn
from models.methods.DSRNet import CCM
from models.module.attention import ChannelAttention, SpatialAttention
from models.module.common import conv3x3_bn_relu
from models.backbone.resnet import ResNet
import torch.nn.functional as F
from timm.layers import DropPath
from timm.layers import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, inc, hidden=None, outc=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        outc = outc or inc
        hidden = hidden or inc
        self.fc1 = nn.Linear(inc, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, outc)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
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


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(inc=dim, hidden=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, depth, num_heads, embed_dim, num_patches, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):

        super(Transformer, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, peb=True):
        # receive x in shape of B,HW,C
        if peb:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

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
        p2 = self.conv_p2(fea.transpose(1, 2).reshape(B, C, self.hw, self.hw))  # pred2

        return [p1, p2, fea]


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


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        set_channels = 64

        # CNN feature channel compression
        self.down1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)
        self.down2 = nn.Conv2d(1024, 320, kernel_size=1, stride=1)
        self.down3 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.down4 = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        # CCM module
        self.ccm_c1 = CCM(512, 256, redio=16)
        self.ccm_c2 = CCM(320, 128, redio=8)
        self.ccm_c3 = CCM(128, 80, redio=4)
        self.ccm_c4 = CCM(64, 32, redio=4)

        # FEM module
        self.fuser7_14 = FEM(emb_dim=320, hw=7, cur_stg=512)
        self.fuser14_28 = FEM(emb_dim=128, hw=14, cur_stg=320)
        self.fuser28_56 = FEM(emb_dim=64, hw=28, cur_stg=128)

        # ERM module
        self.CRM_7 = ERM(inc=512, outc=1024, hw=7, embed_dim=512, num_patches=49)
        self.CRM_14 = ERM(inc=320, outc=256, hw=14, embed_dim=320, num_patches=196)
        self.CRM_28 = ERM(inc=128, outc=64, hw=28, embed_dim=128, num_patches=784)
        self.CRM_56 = ERM(inc=64, outc=16, hw=56, embed_dim=64, num_patches=3136)

    def forward(self, fea, shape):
        # out_7, out_14, out_28, out_56 = torch.rand(1, 49, 512), torch.rand(1, 196, 320), torch.rand(1, 784, 128), torch.rand(1, 3136, 64)
        pred = list()
        # CNN
        c4 = self.down4(fea[0])
        c3 = self.down3(fea[1])
        c2 = self.down2(fea[2])
        c1 = self.down1(fea[3])
        out_56 = c4.flatten(2).transpose(1, 2)
        out_28 = c3.flatten(2).transpose(1, 2)
        out_14 = c2.flatten(2).transpose(1, 2)
        out_7 = c1.flatten(2).transpose(1, 2)

        c4 = self.ccm_c4(c4)  # 56 56 32
        c3 = self.ccm_c3(c3)  # 28 28 80
        c2 = self.ccm_c2(c2)  # 14 14 128
        c1 = self.ccm_c1(c1)  # 7 7 512

        # 1024 * 7 * 7, 1024 * 7 * 7, 49 * 512
        p1_7, p2_7, out_7_ = self.CRM_7(out_7)
        pred.append(F.pixel_shuffle(p1_7, 32))
        pred.append(F.pixel_shuffle(p2_7, 32))

        # 256 * 14 * 14, 256 * 14 * 14, 196 * 320
        out_14_ = self.fuser7_14(out_14, out_7_, c2)  # 196 * 320
        p1_14, p2_14, out_14_ = self.CRM_14(out_14_)
        pred.append(F.pixel_shuffle(p1_14, 16))
        pred.append(F.pixel_shuffle(p2_14, 16))

        # 64 * 28 * 28, 64 * 28 * 28, 784 * 128
        out_28_ = self.fuser14_28(out_28, out_14, c3)  # 784 * 128
        p1_28, p2_28, out_28_ = self.CRM_28(out_28_)
        pred.append(F.pixel_shuffle(p1_28, 8))
        pred.append(F.pixel_shuffle(p2_28, 8))

        # 16 * 56 * 56, 16 * 56 * 56, 3136 * 64
        out_56_ = self.fuser28_56(out_56, out_28, c4)
        p1_56, p2_56, out_56_ = self.CRM_56(out_56_)
        pred.append(F.pixel_shuffle(p1_56, 4))
        pred.append(F.pixel_shuffle(p2_56, 4))
        out = [F.interpolate(x, size=shape, mode='bilinear', align_corners=False) for x in pred]
        return out


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

    input = torch.rand(1, 3, 224, 224)
    edge = torch.rand(2, 1, 56, 56)
    model = PGN()
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    print(ouput[0])
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

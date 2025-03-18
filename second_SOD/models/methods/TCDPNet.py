import math

import torch
import torch.nn as nn
from Transformer import Transformer
import torch.nn.functional as F


# CRM(inc=512, outc=1024, hw=7, embed_dim=512, num_patches=49)
class CRM(nn.Module):
    def __init__(self, inc, outc, hw, embed_dim, num_patches, depth=4):
        super(CRM, self).__init__()
        self.conv_p1 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_p2 = nn.Conv2d(inc, outc, kernel_size=3, padding=1, bias=True)
        self.conv_glb = nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True)

        self.conv_matt = nn.Sequential(nn.Conv2d(outc, inc, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(inc),
                                       nn.LeakyReLU(inplace=True))
        self.conv_fuse = nn.Sequential(nn.Conv2d(2 * inc, inc, kernel_size=3, padding=1, bias=True),
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

    def forward(self, x, glbmap):
        # x in shape of B,N,C
        # glbmap in shape of B,1,224,224
        B, _, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.hw, self.hw)
        if glbmap.shape[-1] // self.hw != 1:
            glbmap = F.pixel_unshuffle(glbmap, glbmap.shape[-1] // self.hw)
            glbmap = self.conv_glb(glbmap)

        x = torch.cat([glbmap, x], dim=1)
        x = self.conv_fuse(x)
        # pred
        p1 = self.conv_p1(x)
        matt = self.sigmoid(p1)
        matt = matt * (1 - matt)
        matt = self.conv_matt(matt)
        fea = x * (1 + matt)

        # reshape x back to B,N,C
        fea = fea.reshape(B, self.inc, -1).transpose(1, 2)
        fea = self.tf(fea, True)
        p2 = self.conv_p2(fea.transpose(1, 2).reshape(B, C, self.hw, self.hw))

        return [p1, p2, fea]


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
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
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


class Fuser(nn.Module):
    def __init__(self, emb_dim=320, hw=7, cur_stg=512):
        super(Fuser, self).__init__()

        self.shuffle = nn.PixelShuffle(2)
        self.unfold = nn.Unfold(kernel_size=1, stride=1, padding=0)
        self.concatFuse = nn.Sequential(nn.Linear(emb_dim + cur_stg // 4, emb_dim),
                                        nn.GELU(),
                                        nn.Linear(emb_dim, emb_dim))
        self.att = Token_performer(dim=emb_dim, in_dim=emb_dim, kernel_ratio=0.5)
        self.hw = hw

    def forward(self, a, b):
        B, _, _ = b.shape
        b = self.shuffle(b.transpose(1, 2).reshape(B, -1, self.hw, self.hw))
        b = self.unfold(b).transpose(1, 2)

        out = self.concatFuse(torch.cat([a, b], dim=2))
        out = self.att(out)

        return out
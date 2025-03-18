import torch
from torch import nn
from timm.models.layers import DropPath, to_2tuple
from models.module.swin import WindowAttention, window_partition, window_reverse


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim1, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim1)

        self.k2 = nn.Linear(dim2, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim2, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, fea, depth_fea):
        _, N1, _ = fea.shape  # [bs, seq_length, channel]
        B, N, _ = depth_fea.shape
        C = self.dim
        q1 = self.q1(fea).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q [B, nhead, N, C // nhead]

        k2 = self.k2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.v2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # k, v [B, nhead, N, C // nhead]
        # 计算注意力得分
        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 应用注意力得分
        fea = (attn @ v2).transpose(1, 2).reshape(B, N1, C)
        fea = self.proj(fea)
        fea = self.proj_drop(fea)

        return fea

    def flops(self, N1, N2):
        flops = 0
        # q
        flops += N1 * self.dim1 * self.dim
        # kv
        flops += N2 * self.dim2 * self.dim * 2
        # qk
        flops += self.num_heads * N1 * self.dim // self.num_heads * N2
        # att v
        flops += self.num_heads * N1 * self.dim // self.num_heads * N2
        # proj
        flops += N1 * self.dim * self.dim1
        return flops


class MultilevelInteractionBlock(nn.Module):
    r""" Multilevel Interaction Block.

    Args:
        dim (int): Number of low-level feature channels.
        dim1, dim2 (int): Number of high-level feature channels.
        embed_dim (int): Dimension for attention.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """

    def __init__(self, dim, dim1, dim2=None, embed_dim=384, num_heads=6, mlp_ratio=3., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MultilevelInteractionBlock, self).__init__()
        self.interact1 = CrossAttention(dim1=dim, dim2=dim1, dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim1)
        self.dim = dim
        self.dim2 = dim2
        self.mlp_ratio = mlp_ratio
        if self.dim2:
            self.interact2 = CrossAttention(dim1=dim, dim2=dim2, dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.norm2 = norm_layer(dim2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            act_layer(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, fea, fea_1, fea_2=None):
        fea = self.norm0(fea)
        fea_1 = self.norm1(fea_1)
        fea_1 = self.interact1(fea, fea_1)
        if self.dim2:
            fea_2 = self.norm2(fea_2)
            fea_2 = self.interact2(fea, fea_2)
        fea = fea + fea_1
        if self.dim2:
            fea = fea + fea_2
        fea = fea + self.drop_path(self.mlp(self.norm(fea)))
        return fea

    def flops(self, N1, N2, N3=None):
        flops = 0
        flops += self.interact1.flops(N1, N2)
        if N3:
            flops += self.interact2.flops(N1, N3)
        flops += self.dim * N1
        flops += 2 * N1 * self.dim * self.dim * self.mlp_ratio
        return flops


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

    def flops(self, N):
        flops = 0
        # q
        flops += N * self.dim * self.dim * 3
        # qk
        flops += self.num_heads * N * self.dim // self.num_heads * N
        # att v
        flops += self.num_heads * N * self.dim // self.num_heads * N
        # proj
        flops += N * self.dim * self.dim
        return flops


class Block(nn.Module):
    # Remove FFN
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.dim = dim
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self, N):
        flops = 0
        # att
        flops += self.attn.flops(N)
        # norm
        flops += self.dim * N
        return flops


class WindowAttentionBlock(nn.Module):
    r""" Based on Swin Transformer Block, We remove FFN.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.norm2 = norm_layer(dim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = self.drop_path(x)

        # FFN
        #x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class MixedAttentionBlock(nn.Module):
    def __init__(self, dim, img_size, window_size, num_heads=1, mlp_ratio=3, drop_path=0.):
        super(MixedAttentionBlock, self).__init__()

        self.img_size = img_size
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.windowatt = WindowAttentionBlock(dim=dim, input_resolution=img_size, num_heads=num_heads,
                                              window_size=window_size, shift_size=0,
                                              mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                                              drop_path=0.,
                                              act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                                              fused_window_process=False)
        self.globalatt = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None,
                               drop=0., attn_drop=0.,
                               drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        att1 = self.windowatt(x)
        att2 = self.globalatt(x)
        x = x + att1 + att2
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x

    def flops(self):
        N = self.img_size[0] * self.img_size[1]
        flops = 0
        flops += self.windowatt.flops()
        flops += self.globalatt.flops(N)
        flops += self.dim * N
        flops += 2 * N * self.dim * self.dim * self.mlp_ratio
        return flops


class multiscale_fusion(nn.Module):
    r""" Upsampling and feature fusion.

    Args:
        in_dim (int): Number of input feature channels.
        f_dim (int): Number of fusion feature channels.
        img_size (int): Image size after upsampling.
        kernel_size (tuple(int)): The size of the sliding blocks.
        stride (int): The stride of the sliding blocks in the input spatial dimensions, can be regarded as upsampling ratio.
        padding (int): Implicit zero padding to be added on both sides of input.
        fuse (bool): If True, concat features from different levels.
    """

    def __init__(self, in_dim, f_dim, kernel_size, img_size, stride, padding, fuse=True):
        super(multiscale_fusion, self).__init__()
        self.fuse = fuse
        self.norm = nn.LayerNorm(in_dim)
        self.in_dim = in_dim
        self.f_dim = f_dim
        self.kernel_size = kernel_size
        self.img_size = img_size
        self.project = nn.Linear(in_dim, in_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=img_size, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.fuse:
            self.mlp1 = nn.Sequential(
                nn.Linear(in_dim + f_dim, f_dim),
                nn.GELU(),
                nn.Linear(f_dim, f_dim),
            )
        else:
            self.proj = nn.Linear(in_dim, f_dim)

    def forward(self, fea, fea_1=None):
        fea = self.project(self.norm(fea))
        fea = self.upsample(fea.transpose(1, 2))
        B, C, _, _ = fea.shape
        fea = fea.view(B, C, -1).transpose(1, 2)  # .contiguous()
        if self.fuse:
            fea = torch.cat([fea, fea_1], dim=2)
            fea = self.mlp1(fea)
        else:
            fea = self.proj(fea)
        return fea

    def flops(self):
        N = self.img_size[0] * self.img_size[1]
        flops = 0
        # norm
        flops += N * self.in_dim
        # proj
        flops += N * self.in_dim * self.in_dim * self.kernel_size[0] * self.kernel_size[1]
        # mlp
        flops += N * (self.in_dim + self.f_dim) * self.f_dim
        flops += N * self.f_dim * self.f_dim
        return flops


class MixedAttention(nn.Module):
    r""" Mixed Attention Module.

    Args:
        in_dim (int): Number of input feature channels.
        dim (int): Number for attention.
        img_size (int): Image size after upsampling.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        depth (int): The number of MAB stacked.
    """

    def __init__(self, in_dim, dim, img_size, window_size, num_heads=1, mlp_ratio=4, depth=2, drop_path=0.):
        super(MixedAttention, self).__init__()

        self.img_size = img_size
        self.in_dim = in_dim
        self.dim = dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([
            MixedAttentionBlock(dim=dim, img_size=img_size, window_size=window_size, num_heads=num_heads,
                                mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, fea):
        fea = self.mlp1(self.norm1(fea))
        for blk in self.blocks:
            fea = blk(fea)
        fea = self.drop_path(self.mlp2(self.norm2(fea)))
        return fea

    def flops(self):
        flops = 0
        N = self.img_size[0] * self.img_size[1]
        # norm1
        flops += N * self.in_dim
        # mlp1
        flops += N * self.in_dim * self.dim
        flops += N * self.dim * self.dim
        # blks
        for blk in self.blocks:
            flops += blk.flops()
        # norm2
        flops += N * self.dim
        # mlp2
        flops += N * self.in_dim * self.dim
        flops += N * self.dim * self.dim
        return flops


class decoder(nn.Module):
    r""" Multistage decoder.

    Args:
        embed_dim (int): Dimension for attention. Default 384
        dim (int): Patch embedding dimension. Default 96
        img_size (int): Input image size. Default 224
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """

    def __init__(self, embed_dim=384, dims=[96, 192, 384], img_size=224, mlp_ratio=3):
        super(decoder, self).__init__()
        self.img_size = img_size
        self.dims = dims
        self.embed_dim = embed_dim
        self.fusion1 = multiscale_fusion(in_dim=dims[2], f_dim=dims[1], kernel_size=(3, 3),
                                         img_size=(img_size // 8, img_size // 8), stride=(2, 2), padding=(1, 1))
        self.fusion2 = multiscale_fusion(in_dim=dims[1], f_dim=dims[0], kernel_size=(3, 3),
                                         img_size=(img_size // 4, img_size // 4), stride=(2, 2), padding=(1, 1))
        self.fusion3 = multiscale_fusion(in_dim=dims[0], f_dim=dims[0], kernel_size=(7, 7),
                                         img_size=(img_size // 1, img_size // 1), stride=(4, 4), padding=(2, 2),
                                         fuse=False)

        self.mixatt1 = MixedAttention(in_dim=dims[1], dim=embed_dim, img_size=(img_size // 8, img_size // 8),
                                      window_size=(img_size // 32), num_heads=1, mlp_ratio=mlp_ratio, depth=2)
        self.mixatt2 = MixedAttention(in_dim=dims[0], dim=embed_dim, img_size=(img_size // 4, img_size // 4),
                                      window_size=(img_size // 32), num_heads=1, mlp_ratio=mlp_ratio, depth=2)

        self.proj1 = nn.Linear(dims[2], 1)
        self.proj2 = nn.Linear(dims[1], 1)
        self.proj3 = nn.Linear(dims[0], 1)
        self.proj4 = nn.Linear(dims[0], 1)

    def forward(self, f):
        fea_1_16, fea_1_8, fea_1_4 = f  # fea_1_16:1/16
        B, _, _ = fea_1_16.shape
        fea_1_8 = self.fusion1(fea_1_16, fea_1_8)
        fea_1_8 = self.mixatt1(fea_1_8)

        fea_1_4 = self.fusion2(fea_1_8, fea_1_4)
        fea_1_4 = self.mixatt2(fea_1_4)

        fea_1_1 = self.fusion3(fea_1_4)

        fea_1_16 = self.proj1(fea_1_16)
        mask_1_16 = fea_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)
        fea_1_8 = self.proj2(fea_1_8)
        mask_1_8 = fea_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)
        fea_1_4 = self.proj3(fea_1_4)
        mask_1_4 = fea_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)
        fea_1_1 = self.proj4(fea_1_1)
        mask_1_1 = fea_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)
        return [mask_1_16, mask_1_8, mask_1_4, mask_1_1]

    def flops(self):
        flops = 0
        flops += self.fusion1.flops()
        flops += self.fusion2.flops()
        flops += self.fusion3.flops()
        flops += self.mixatt1.flops()
        flops += self.mixatt2.flops()

        flops += self.img_size // 16 * self.img_size // 16 * self.dims[2]
        flops += self.img_size // 8 * self.img_size // 8 * self.dims[1]
        flops += self.img_size // 4 * self.img_size // 4 * self.dims[0]
        flops += self.img_size // 1 * self.img_size // 1 * self.dims[0]

        return flops


class MAB(nn.Module):
    def __init__(self, embed_dim=384, dims=[96, 192, 384, 768], img_size=224, mlp_ratio=3):
        super(MAB, self).__init__()
        self.img_size = img_size
        self.dims = dims
        self.embed_dim = embed_dim
        self.fusion4 = multiscale_fusion(in_dim=dims[3], f_dim=dims[2], kernel_size=(3, 3),
                                         img_size=img_size // 16, stride=2, padding=1)
        self.fusion3 = multiscale_fusion(in_dim=dims[2], f_dim=dims[1], kernel_size=(3, 3),
                                         img_size=img_size // 8, stride=2, padding=1)
        self.fusion2 = multiscale_fusion(in_dim=dims[1], f_dim=dims[0], kernel_size=(3, 3),
                                         img_size=img_size // 4, stride=2, padding=1)
        self.fusion1 = multiscale_fusion(in_dim=dims[0], f_dim=dims[0], kernel_size=(7, 7),
                                         img_size=img_size // 1, stride=4, padding=2,
                                         fuse=False)

        self.mixatt1 = MixedAttention(in_dim=dims[0], dim=embed_dim, img_size=(img_size // 4, img_size // 4),
                                      window_size=(img_size // 32), num_heads=1, mlp_ratio=mlp_ratio, depth=2)
        self.mixatt2 = MixedAttention(in_dim=dims[1], dim=embed_dim, img_size=(img_size // 8, img_size // 8),
                                      window_size=(img_size // 32), num_heads=1, mlp_ratio=mlp_ratio, depth=2)
        self.mixatt3 = MixedAttention(in_dim=dims[2], dim=embed_dim, img_size=(img_size // 16, img_size // 16),
                                      window_size=(img_size // 32), num_heads=1, mlp_ratio=mlp_ratio, depth=2)

        self.proj4 = nn.Linear(dims[3], 1)
        self.proj3 = nn.Linear(dims[2], 1)
        self.proj2 = nn.Linear(dims[1], 1)
        self.proj1 = nn.Linear(dims[0], 1)
        self.proj0 = nn.Linear(dims[0], 1)

    def forward(self, fea):
        fea_1, fea_2, fea_3, fea_4 = fea
        B, C, _ = fea_4.shape
        fea_3 = self.fusion4(fea_4, fea_3)
        fea_3 = self.mixatt3(fea_3)

        fea_2 = self.fusion3(fea_3, fea_2)
        fea_2 = self.mixatt2(fea_2)

        fea_1 = self.fusion2(fea_2, fea_1)
        fea_1 = self.mixatt1(fea_1)

        fea_0 = self.fusion1(fea_1)

        fea_4 = self.proj4(fea_4)
        mask_4 = fea_4.transpose(1, 2).reshape(B, 1, self.img_size // 32, self.img_size // 32)

        fea_3 = self.proj3(fea_3)
        mask_3 = fea_3.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)

        fea_2 = self.proj2(fea_2)
        mask_2 = fea_2.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)

        fea_1 = self.proj1(fea_1)
        mask_1 = fea_1.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)

        fea_0 = self.proj0(fea_0)
        mask_0 = fea_0.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)
        return mask_0, mask_1, mask_2, mask_3, mask_4


if __name__ == '__main__':
    f1 = torch.rand(1, 56 * 56, 64)
    f2 = torch.rand(1, 28 * 28, 64)
    f3 = torch.rand(1, 14 * 14, 64)
    f4 = torch.rand(1, 7 * 7, 64)
    net = MAB(dims=[64, 64, 64, 64])
    out = net((f1, f2, f3, f4))
    for i in range(5):
        print(out[i].shape)
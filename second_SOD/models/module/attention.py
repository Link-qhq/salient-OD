import torch
from torch import nn
from models.module.common import conv3x3_bn_silu
from models.module.common import weight_init
from einops import rearrange


# Self-Attention
class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CAAttention(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CAAttention, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global maximum pooling

        # Channel compression and recovery
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.initialize()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # Global average pooling
        max_out = self.fc(self.max_pool(x))  # Global maximum pooling
        out = avg_out + max_out  # Two kinds of information merge
        # sigmoid is used for empowerment
        return self.sigmoid(out)

    def initialize(self):
        weight_init(self)


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.initialize()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], 1)
        x = self.conv1(x)
        return self.sigmoid(x)

    def initialize(self):
        weight_init(self)


class SpaceAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpaceAttention, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self._x = nn.Conv2d(2, 1, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, kernel_size // 2))
        self._y = nn.Conv2d(2, 1, kernel_size=(kernel_size, 1), stride=(1, 1), padding=(kernel_size // 2, 0))
        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.initialize()

    def forward(self, x):
        x1 = self.avg(x)
        x2 = self.max(x)
        x = torch.cat([x1, x2], 1)
        return self.sigmoid(self.conv(torch.cat([self._x(x), self._y(x)], 1)))

    def initialize(self):
        weight_init(self)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=4, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)
        self.initialize()

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

    def initialize(self):
        weight_init(self)


class ResCBAM(nn.Module):
    def __init__(self, in_planes, reduction=4, kernel_size=7):
        super(ResCBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result + x


class SegNext_Attention(nn.Module):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 1, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (5, 1), padding=(2, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        # return attn
        return attn * u


class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = conv3x3_bn_silu(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = conv3x3_bn_silu(ch, ch)
        self.act = nn.Sigmoid()
        self.initialize()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x

    def initialize(self):
        weight_init(self)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
        self.initialize()

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        pass


class RFCAConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, reduction=32):
        super(RFCAConv, self).__init__()
        self.kernel_size = kernel_size
        self.generate = nn.Sequential(nn.Conv2d(inp, inp * (kernel_size ** 2), kernel_size, padding=kernel_size // 2,
                                                stride=stride, groups=inp,
                                                bias=False),
                                      nn.BatchNorm2d(inp * (kernel_size ** 2)),
                                      nn.ReLU()
                                      )
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(nn.Conv2d(inp, oup, kernel_size, stride=kernel_size))

    def forward(self, x):
        b, c = x.shape[0:2]
        generate_feature = self.generate(x)
        h, w = generate_feature.shape[2:]
        generate_feature = generate_feature.view(b, c, self.kernel_size ** 2, h, w)

        generate_feature = rearrange(generate_feature, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                                     n2=self.kernel_size)

        x_h = self.pool_h(generate_feature)
        x_w = self.pool_w(generate_feature).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        h, w = generate_feature.shape[2:]
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return self.conv(generate_feature * a_w * a_h)


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out


class CA(nn.Module):
    def __init__(self, inp, reduction=16):
        super(CA, self).__init__()
        # h:height(行)   w:width(列)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)

        # mip = max(8, inp // reduction)  # 论文作者所用
        mip = inp // reduction  # 博主所用   reduction = int(math.sqrt(inp))

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

    def initialize(self):
        weight_init(self)


# Attention Block
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9574645
# Low-altitude small-sized object detection using lightweight feature-enhanced convolutional neural network
# 由SE和CA注意力机制启发
class AttentionBlock(nn.Module):
    def __init__(self, channel, ratio=0.25):
        super(AttentionBlock, self).__init__()
        hidden_dim = int(channel * ratio)
        self.global_avgPool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 2, 1, 1),
        )
        self.global_maxPool = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channel, channel // 2, 1, 1)
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, 1, 1), nn.BatchNorm2d(channel // 2), nn.ReLU(True),
            nn.Conv2d(channel // 2, channel // 4, 1, 1), nn.BatchNorm2d(channel // 4), nn.ReLU(True),
            nn.Conv2d(channel // 4, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU(True),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, channel, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cat = torch.cat([self.global_maxPool(x), self.global_avgPool(x)], 1)
        cat = cat + self.conv_res(cat)
        return x * self.conv_out(cat)


class SCSA_SA(nn.Module):  # spatial_size
    def __init__(self, dim, group_kernel_sizes=None, gate_layer='softmax'):
        super(SCSA_SA, self).__init__()
        if group_kernel_sizes is None:
            group_kernel_sizes = [3, 5, 7, 9]
        self.dim = dim
        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

    def forward(self, x):
        # Spatial attention priority calculation
        b, c, h_, w_ = x.shape
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        return x_h_attn * x_w_attn


class SCSA_CA(nn.Module):  # channel self-attention
    def __init__(self,
                 dim,
                 head_num=8,
                 window_size=7,
                 down_sample_mode='avg_pool',
                 qkv_bias=False,
                 attn_drop_ratio=0.,
                 gate_layer='softmax'
                 ):
        super(SCSA_CA, self).__init__()
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

    def forward(self, x):
        # Channel attention based on self attention
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn


class SCSA(nn.Module):

    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes=None,
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(SCSA, self).__init__()
        if group_kernel_sizes is None:
            group_kernel_sizes = [3, 5, 7, 9]
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dim of x is (B, C, H, W)
        """
        # Spatial attention priority calculation
        b, c, h_, w_ = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn

        # Channel attention based on self attention
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)
        return attn * x


if __name__ == '__main__':
    x = torch.randn(1, 64, 14, 14)  # b, c, h, w
    net = SCSA_SA(64)
    print(net(x).shape)

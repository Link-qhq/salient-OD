import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.attention import ChannelAttention, SpatialAttention
from models.module.common import conv3x3_bn_relu


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

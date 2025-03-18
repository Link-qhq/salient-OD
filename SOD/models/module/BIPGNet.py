import torch
from torch import nn
from torch.nn import functional as F
from models.module.DSRNet import CCM
from models.module.Transformer import Mlp


class BIG(nn.Module):
    """
        boundary information guidance 边界信息引导模块
    """

    def __init__(self, channel, groups=16):
        super(BIG, self).__init__()
        self.gate1 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())
        self.gate4 = nn.Sequential(nn.Conv2d(channel // 4, channel // 4, kernel_size=3, padding=1),
                                   nn.GroupNorm(groups, channel // 4), nn.PReLU(),
                                   nn.Conv2d(channel // 4, 1, kernel_size=1),
                                   nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                                  nn.GroupNorm(groups, channel), nn.PReLU())

        self.channel = channel
        self.weight = nn.Softmax(dim=1)

    def forward(self, x, edge):
        """
        :param x: torch.Size([1, channel(↑), 56, 56])
        :param edge: torch.Size([1, 1, 56, 56])
        :return:
        """
        x1, x2, x3, x4 = torch.split(x, self.channel // 4, dim=1)

        cm1 = self.gate1(x1)
        cm2 = self.gate2(x2)
        cm3 = self.gate3(x3)
        cm4 = self.gate4(x4)

        e1 = cm1 * torch.sigmoid(edge)
        e2 = cm2 * torch.sigmoid(edge)
        e3 = cm3 * torch.sigmoid(edge)
        e4 = cm4 * torch.sigmoid(edge)

        gv1 = F.avg_pool2d(e1, (e1.size(2), e1.size(3)), stride=(e1.size(2), e1.size(3)))
        gv2 = F.avg_pool2d(e2, (e2.size(2), e2.size(3)), stride=(e2.size(2), e2.size(3)))
        gv3 = F.avg_pool2d(e3, (e3.size(2), e3.size(3)), stride=(e3.size(2), e3.size(3)))
        gv4 = F.avg_pool2d(e4, (e4.size(2), e4.size(3)), stride=(e4.size(2), e4.size(3)))

        weight = self.weight(torch.cat((gv1, gv2, gv3, gv4), 1))
        w1, w2, w3, w4 = torch.split(weight, 1, dim=1)

        nx1 = x1 * w1
        nx2 = x2 * w2
        nx3 = x3 * w3
        nx4 = x4 * w4

        return self.conv(torch.cat((nx1, nx2, nx3, nx4), 1))


class SDU(nn.Module):
    def __init__(self, channel=256, groups=32):
        super(SDU, self).__init__()
        if channel % groups != 0:
            assert 'groups must be divisible channel'
        # SDU
        self.pr1_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(groups, channel), nn.PReLU()
        )
        self.pr1_2 = nn.Sequential(
            CCM(channel, channel, 16),
            nn.Conv2d(channel, 1, kernel_size=1)
        )
        self.pe1_1 = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=3, padding=1), nn.GroupNorm(groups, channel), nn.PReLU(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.GroupNorm(groups, channel), nn.PReLU()
        )
        self.pe1_2 = nn.Sequential(
            # CCM(channel, channel, 16),
            nn.Conv2d(channel, 1, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.pr1_1(x)
        out2 = self.pr1_2(out1)
        out3 = self.pe1_1(out2)
        out4 = self.pe1_2(out3)
        return out1, out2, out3, out4

if __name__ == '__main__':
    input1 = torch.rand(1, 64, 224, 224)
    input2 = torch.rand(1, 64, 56, 56)
    input3 = torch.rand(1, 64, 112, 112)
    ice = BIG(64)
    # ouput = ice(input1, input2)
    # print(ouput.shape)
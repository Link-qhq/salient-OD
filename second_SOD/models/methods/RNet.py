import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import time
import os
import torch.nn.functional as F
from models.backbone.Rresnet import B2_ResNet


class HA_r(nn.Module):
    def __init__(self, channels):
        super(HA_r, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, attention, x):
        x = self.conv(x * attention) + x
        return x


class C_attention(nn.Module):

    def __init__(self, channels=32):
        super(C_attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c).contiguous()
        y = self.fc(y).view(b, c, 1, 1).contiguous()
        x = torch.mul(y, x)
        x = self.out_conv(x)
        return x


class Self_Attention(nn.Module):
    def __init__(self, feature_size=88):
        super(Self_Attention, self).__init__()
        self.pooling = nn.AdaptiveAvgPool3d((1, feature_size, feature_size))
        self.linear = nn.Sequential(nn.Linear(feature_size ** 2, (feature_size ** 2) // 8),
                                    nn.ReLU(inplace=True),
                                    nn.Linear((feature_size ** 2) // 8, feature_size ** 2),
                                    nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, 1, c, h, w).contiguous()
        x = self.pooling(x).view(b, h * w).contiguous()
        x = self.linear(x)
        x = x.view(b, 1, h, w)
        return x


class dense_aggregation(nn.Module):

    def __init__(self, inchannels=32, feature_num=4, up_num=2):
        super(dense_aggregation, self).__init__()
        self.conv_upsample = []
        for i in range(up_num):
            self.conv_upsample.append(
                nn.Sequential(nn.Conv2d(inchannels * (i + 1), inchannels * (i + 1), (3, 3), padding=1),
                              nn.ReLU(inplace=True)))
        self.conv_upsample1 = nn.ModuleList(self.conv_upsample)

        self.conv_poolsample = []
        for i in range(feature_num - up_num - 1):
            self.conv_poolsample.append(
                nn.Sequential(nn.Conv2d(inchannels * (i + 1), inchannels * (i + 1), (3, 3), padding=1, stride=(2, 2),
                                        groups=inchannels * (i + 1)),
                              nn.Conv2d(inchannels * (i + 1), inchannels * (i + 1), (3, 3), padding=1),
                              nn.ReLU(inplace=True)))
        self.conv_poolsample1 = nn.ModuleList(self.conv_poolsample)

        self.upconv = nn.Conv2d(inchannels, inchannels, 3, padding=1)
        self.c_attention = C_attention(inchannels * feature_num)
        self.conv = nn.Sequential(
            nn.Conv2d(inchannels * feature_num, inchannels * feature_num, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(inchannels * feature_num),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels=inchannels * feature_num, out_channels=inchannels, kernel_size=(1, 1)),
            nn.BatchNorm2d(inchannels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inchannels, inchannels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True))

        self.feature_num = feature_num

    def forward(self, pool_x, x, up_x, mask):
        counter = 0

        while len(up_x) >= 2:
            x1 = up_x.pop(-1)
            x1 = F.interpolate(x1, size=up_x[-1].size()[2:], mode='bilinear', align_corners=False)
            x1 = self.conv_upsample1[counter](x1)
            up_x[-1] = torch.cat((x1, up_x[-1]), dim=1)
            counter += 1
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear', align_corners=False)
        mask = self.upconv(mask)
        mask = torch.cat(tuple([mask for i in range(self.feature_num)]), dim=1)
        counter = 0
        while len(pool_x) >= 2:
            x1 = pool_x.pop(0)
            x1 = self.conv_poolsample1[counter](x1)
            pool_x[0] = torch.cat((x1, pool_x[0]), dim=1)
            counter += 1
        if len(pool_x) == 1 and len(up_x) == 1:
            up_x[0] = F.interpolate(up_x[0], x.size()[2:], mode='bilinear', align_corners=False)
            x_sum = torch.cat((self.conv_poolsample1[-1](pool_x[0]),
                               x,
                               self.conv_upsample1[-1](up_x[0])), dim=1)
        elif len(pool_x) == 0:
            up_x[0] = F.interpolate(up_x[0], x.size()[2:], mode='bilinear', align_corners=False)
            x_sum = torch.cat((x, self.conv_upsample1[-1](up_x[0])), dim=1)
        else:
            x_sum = torch.cat((self.conv_poolsample1[-1](pool_x[0]), x), dim=1)
        x_sum = x_sum * mask
        x_sum = self.c_attention(x_sum)
        x = self.conv(x_sum)
        return x


class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        maps = {2048: 11, 1024: 22, 512: 44, 256: 88}
        self.self_attention = Self_Attention(maps[in_channel])
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1)),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, (3, 3), padding=3, dilation=(3, 3))
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, (3, 3), padding=5, dilation=(5, 5))
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, (3, 3), padding=7, dilation=(7, 7))
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, (3, 3), padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, (1, 1))

    def forward(self, x):
        attention = self.self_attention(x)
        x = x * attention
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))

        return x



class my_aggregation(nn.Module):

    def __init__(self, inchannels=32):
        super(my_aggregation, self).__init__()
        self.dense_agg1 = dense_aggregation(inchannels, feature_num=4, up_num=1)
        self.dense_agg2 = dense_aggregation(inchannels, feature_num=4, up_num=2)
        self.dense_agg3 = dense_aggregation(inchannels, feature_num=4, up_num=3)
        self.outConv = nn.Sequential(nn.Conv2d(inchannels, inchannels, (3, 3), padding=1),
                                     nn.ReLU6(inplace=True),
                                     nn.Conv2d(inchannels, 1, (1, 1)))

    def forward(self, x2, x3, x4, x5):
        mask = x5
        merge = self.dense_agg1([x2, x3], x4, [x5], mask)
        merge = self.dense_agg2([x2], x3, [x4, x5], merge)
        merge = self.dense_agg3([], x2, [x3, x4, x5], merge)
        merge = self.outConv(merge)
        return merge


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


class R_RES(nn.Module):
    def __init__(self, channel=32):
        super(R_RES, self).__init__()
        self.save_mid = False
        self.resnet = B2_ResNet()
        self.head1_rfb1_1 = RFB(256, channel)
        self.head1_rfb2_1 = RFB(512, channel)
        self.head1_rfb3_1 = RFB(1024, channel)
        self.head1_rfb4_1 = RFB(2048, channel)
        self.head1_agg1 = my_aggregation(channel)

        self.head2_rfb1_2 = RFB(256, channel)
        self.head2_rfb2_2 = RFB(512, channel)
        self.head2_rfb3_2 = RFB(1024, channel)
        self.head2_rfb4_2 = RFB(2048, channel)
        self.head2_agg2 = my_aggregation(channel)

        self.head2_HA3 = HA_r(512)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.refine1 = ReFine(256)
        self.refine2 = ReFine(256)
        self.pooling2 = nn.MaxPool2d(kernel_size=(2, 2))
        if self.training:
            self.initialize_weights()

    def forward(self, x, shape=None):  # 64x352x352
        x0 = self.resnet.conv1(x)  # 64x156x156
        x0 = self.resnet.bn1(x0)  # 64x156x156
        x0 = self.resnet.relu(x0)  # 64x156x156
        x_refine1 = self.resnet.layer1_1(x0)  # 64x156x156
        x_refine2 = x_refine1
        x1 = self.resnet.maxpool(x0)  # 64x88x88
        x1 = self.resnet.layer1(x1)  # 256 x 88 x 88
        x2 = self.resnet.layer2(x1)  # 512 x 44 x 44

        x2_1 = x2
        x3_1 = self.resnet.layer3_1(x2_1)  # 1024 x 22 x 22
        x4_1 = self.resnet.layer4_1(x3_1)  # 2048 x 11 x 11
        x1_1 = self.head1_rfb1_1(x1)
        x2_1 = self.head1_rfb2_1(x2_1)
        x3_1 = self.head1_rfb3_1(x3_1)
        x4_1 = self.head1_rfb4_1(x4_1)
        attention = self.head1_agg1(x1_1, x2_1, x3_1, x4_1)

        x2_2 = self.head2_HA3(self.pooling2(attention).sigmoid(), x2)
        x3_2 = self.resnet.layer3_2(x2_2)  # 1024 x 22 x 22
        x4_2 = self.resnet.layer4_2(x3_2)  # 2048 x 11 x 11
        x1_2 = self.head2_rfb1_2(x1)
        x2_2 = self.head2_rfb2_2(x2_2)
        x3_2 = self.head2_rfb3_2(x3_2)
        x4_2 = self.head2_rfb4_2(x4_2)
        detection = self.head2_agg2(x1_2, x2_2, x3_2, x4_2)

        attention = self.upsample(attention)

        attention2 = self.upsample(self.refine1(x_refine1, attention))

        detection = self.upsample(detection)
        detection2 = self.upsample(self.refine2(x_refine2, detection))
        return attention2, detection2

    def initialize_weights(self):
        self.load_state_dict(
            torch.load('/home/amax/文档/qhq/second_SOD/pretrained_model/resnet50.pth', weights_only=True), strict=False)
        print("\033[94mPre-trained ResNet weight loaded.\033[0m")
        # res50 = models.resnet50(pretrained=True)
        # pretrained_dict = res50.state_dict()
        # all_params = {}
        # for k, v in self.resnet.state_dict().items():
        #     if k in pretrained_dict.keys():
        #         v = pretrained_dict[k]
        #         all_params[k] = v
        #     elif '_1' in k:
        #         name = k.split('_1')[0] + k.split('_1')[1]
        #         v = pretrained_dict[name]
        #         all_params[k] = v
        #     elif '_2' in k:
        #         name = k.split('_2')[0] + k.split('_2')[1]
        #         v = pretrained_dict[name]
        #         all_params[k] = v
        # assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        # self.resnet.load_state_dict(all_params)


if __name__ == '__main__':
    from thop import profile

    input = torch.rand(2, 3, 352, 352)
    edge = torch.rand(2, 1, 56, 56)
    model = R_RES()
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
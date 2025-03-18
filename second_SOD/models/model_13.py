import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.common import ConvBNReLU, ReceptiveConv
from models.backbone.resnet_2 import Bottleneck, resnet50


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EDN(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True, use_carafe=True, freeze_s1=False):
        super(EDN, self).__init__()
        self.arch = arch
        self.encoder = resnet50(pretrained=True)

        enc_channels = [64, 256, 512, 1024, 2048, 1024, 1024]
        dec_channels = [32, 64, 256, 512, 512, 128, 128]

        use_dwconv = 'mobilenet' in arch

        self.inplanes = enc_channels[-3]
        self.base_width = 64
        self.conv6 = nn.Sequential(
            self._make_layer(enc_channels[-2] // 4, 2, stride=2),
        )
        self.conv7 = nn.Sequential(
            self._make_layer(enc_channels[-2] // 4, 2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fpn = CustomDecoder(enc_channels, dec_channels, use_dwconv=use_dwconv)

        self.cls1 = nn.Conv2d(dec_channels[0], 1, 1, stride=1, padding=0)
        self.cls2 = nn.Conv2d(dec_channels[1], 1, 1, stride=1, padding=0)
        self.cls3 = nn.Conv2d(dec_channels[2], 1, 1, stride=1, padding=0)
        self.cls4 = nn.Conv2d(dec_channels[3], 1, 1, stride=1, padding=0)
        self.cls5 = nn.Conv2d(dec_channels[4], 1, 1, stride=1, padding=0)
        # self._freeze_backbone(freeze_s1=freeze_s1)

    def _make_layer(self, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        groups = 1
        expansion = 4
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, groups,
                             self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, groups=groups,
                                     base_width=self.base_width, dilation=1,
                                     norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _freeze_backbone(self, freeze_s1):
        if not freeze_s1:
            return
        assert ('resnet' in self.arch and '3x3' not in self.arch)
        m = [self.encoder.conv1, self.encoder.bn1, self.encoder.relu]
        print("freeze stage 0 of resnet")
        for p in m:
            if p in p.parameters():
                p.requires_grad = False

    def forward(self, input):

        backbone_features = self.encoder(input)

        ed1 = self.conv6(backbone_features[-1])
        ed2 = self.conv7(ed1)
        attention = torch.sigmoid(self.gap(ed2))

        features = self.fpn(backbone_features + [ed1, ed2], attention)

        saliency_maps = []
        for idx, feature in enumerate(features[:5]):
            saliency_maps.append(F.interpolate(
                getattr(self, 'cls' + str(idx + 1))(feature),
                input.shape[2:],
                mode='bilinear',
                align_corners=False)
            )
            # p2t can alternatively use features of 4 levels. Here 5 levels are applied.

        return saliency_maps


class CustomDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, use_dwconv=False):
        super(CustomDecoder, self).__init__()
        self.inners_a = nn.ModuleList()
        self.inners_b = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.inners_a.append(ConvBNReLU(in_channels[i], out_channels[i] // 2, ksize=1, pad=0))
            self.inners_b.append(ConvBNReLU(out_channels[i + 1], out_channels[i] // 2, ksize=1, pad=0))
        self.inners_a.append(ConvBNReLU(in_channels[-1], out_channels[-1], ksize=1, pad=0))

        self.fuse = nn.ModuleList()
        dilation = [[1, 2, 4, 8]] * (len(in_channels) - 4) + [[1, 2, 3, 4]] * 2 + [[1, 1, 1, 1]] * 2
        baseWidth = [32] * (len(in_channels) - 5) + [24] * 5
        print("using dwconv:", use_dwconv)
        for i in range(len(in_channels)):
            self.fuse.append(nn.Sequential(
                ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i],
                              use_dwconv=use_dwconv),
                ReceptiveConv(out_channels[i], out_channels[i], baseWidth=baseWidth[i], dilation=dilation[i],
                              use_dwconv=use_dwconv)))

    def forward(self, features, att=None):
        if att is not None:
            stage_result = self.fuse[-1](self.inners_a[-1](features[-1] * att))
        else:
            stage_result = self.fuse[-1](self.inners_a[-1](features[-1]))
        results = [stage_result]
        num_mul_att = 1
        for idx in range(len(features) - 2, -1, -1):
            inner_top_down = F.interpolate(self.inners_b[idx](stage_result),
                                           size=features[idx].shape[2:],
                                           mode='bilinear',
                                           align_corners=False)
            if att is not None and att.shape[1] == features[idx].shape[1] and num_mul_att:
                features[idx] = features[idx] * att
                num_mul_att -= 1
            inner_lateral = self.inners_a[idx](features[idx])
            stage_result = self.fuse[idx](torch.cat((inner_top_down, inner_lateral), dim=1))
            results.insert(0, stage_result)

        return results


if __name__ == '__main__':
    from thop import profile

    input = torch.rand(2, 3, 224, 224)
    edge = torch.rand(2, 1, 56, 56)
    model = EDN(arch='resnet50', pretrained=False, freeze_s1=True)
    ouput = model(input)
    # for i in range(8):
    #     print(ouput[i].shape)
    print(ouput[0])
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
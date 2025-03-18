import torch
from torch import nn
from models.module.resnet import ResNet
from models.module.ICONet import ICE, DFA
import torch.nn.functional as F
from models.module.BIPGNet import SDU, BIG
from models.module.attention import CBAM
from models.module.CSEPNet import CSFI
# from models.module.upgrade import DFA


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        set_channels = 128
        self.dfa0 = DFA(64, set_channels)
        self.dfa1 = DFA(256, set_channels)
        self.dfa2 = DFA(512, set_channels)
        self.dfa3 = DFA(1024, set_channels)
        self.dfa4 = DFA(2048, set_channels)

        self.cbam1 = CBAM(256, 16, 7)
        self.cbam2 = CBAM(512, 16, 7)
        self.cbam3 = CBAM(1024, 16, 7)
        self.cbam4 = CBAM(2048, 16, 7)

        self.ice1 = ICE(set_channels)
        self.ice2 = ICE(set_channels)
        self.ice3 = ICE(set_channels)
        self.ice4 = ICE(set_channels)

        self.big0 = BIG(set_channels)
        self.big1 = BIG(set_channels)
        self.big2 = BIG(set_channels)
        self.big3 = BIG(set_channels)
        self.csfi1 = CSFI(set_channels, set_channels // 2)
        self.csfi2 = CSFI(set_channels, set_channels // 2)
        self.csfi3 = CSFI(set_channels, set_channels // 2)
        self.sdu0 = SDU(set_channels, set_channels // 4)
        self.sdu1 = SDU(set_channels, set_channels // 4)
        self.sdu2 = SDU(set_channels, set_channels // 4)
        self.sdu3 = SDU(set_channels, set_channels // 4)
        self.sdu4 = SDU(set_channels, set_channels // 4)
        # self.linear = nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Sequential(
            CSFI(set_channels, set_channels // 2),
            nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.linear2 = nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, fea, shape):
        fea_1, fea_2, fea_3, fea_4 = fea
        fea_1 = self.dfa1(fea_1)
        fea_2 = self.dfa2(fea_2)
        fea_3 = self.dfa3(fea_3)
        fea_4 = self.dfa4(fea_4)

        out4_1, out4_2, out4_3, out4_4 = self.sdu4(fea_4)

        out3 = self.ice4(fea_3, out4_1, out4_3)
        out3 = self.big3(out3, F.interpolate(out4_4, size=out3.shape[2:], mode="bilinear"))
        out3 = self.csfi3(out3)

        out3_1, out3_2, out3_3, out3_4 = self.sdu3(out3)

        out2 = self.ice3(fea_2, out3_1, out3_3)
        out2 = self.big2(out2, F.interpolate(out3_4, size=out2.shape[2:], mode="bilinear"))
        out2 = self.csfi2(out2)

        out2_1, out2_2, out2_3, out2_4 = self.sdu2(out2)

        out1 = self.ice2(fea_1, out2_1, out2_3)
        out1 = self.big1(out1, F.interpolate(out2_4, size=out1.shape[2:], mode="bilinear"))
        out1 = self.csfi1(out1)

        out1_1, out1_2, out1_3, out1_4 = self.sdu1(out1)

        out1 = F.interpolate(out1_2, size=shape, mode="bilinear", align_corners=True)
        out2 = F.interpolate(out2_2, size=shape, mode="bilinear", align_corners=True)
        out3 = F.interpolate(out3_2, size=shape, mode="bilinear", align_corners=True)
        out4 = F.interpolate(out4_2, size=shape, mode="bilinear", align_corners=True)

        out1_1 = F.interpolate(out1_4, size=shape, mode="bilinear", align_corners=True)
        out2_1 = F.interpolate(out2_4, size=shape, mode="bilinear", align_corners=True)
        out3_1 = F.interpolate(out3_4, size=shape, mode="bilinear", align_corners=True)
        out4_1 = F.interpolate(out4_4, size=shape, mode="bilinear", align_corners=True)

        return out4, out3, out2, out1, out4_1, out3_1, out2_1, out1_1


class TestMODEL(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_size=224):
        super(TestMODEL, self).__init__()
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
    model = TestMODEL(3, 1)
    ouput = model(input)
    model.load_state_dict(torch.load('/root/autodl-tmp/SOD/results/exp_32-BEST/Model_9.pth'))
    # for i in range(8):
    #     print(ouput[i].shape)
    # flops, params = profile(model, inputs=(input,))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')

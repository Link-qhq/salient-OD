import torch
from torch import nn
from models.module.resnet import ResNet
from models.module.ICONet import DFA, ICE
import torch.nn.functional as F
from models.module.BIPGNet import SDU, BIG
from models.module.FP import iAFF
from models.module.SPP import SPPF_LSKA
from models.module.attention import CBAM, CA_Block
from models.module.M3Net import MultilevelInteractionBlock
from models.module.CSEPNet import CSFI, CSC


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        set_channels = 64
        size = 256
        self.dfa1 = DFA(256, set_channels)
        self.dfa2 = DFA(512, set_channels)
        self.dfa3 = DFA(1024, set_channels)
        self.dfa4 = DFA(2048, set_channels)

        # self.cbam1 = CA_Block(256, size // 4, size // 4)
        # self.cbam2 = CA_Block(512, size // 8, size // 8)
        # self.cbam3 = CA_Block(1024, size // 16, size // 16)
        # self.cbam4 = CA_Block(2048, size // 32, size // 32)
        self.cbam1 = CBAM(256, 16, 7)
        self.cbam2 = CBAM(512, 16, 7)
        self.cbam3 = CBAM(1024, 16, 7)
        self.cbam4 = CBAM(2048, 16, 7)
        self.sppf = SPPF_LSKA(2048, set_channels)

        self.ice1 = ICE(set_channels)
        self.ice2 = ICE(set_channels)

        self.fusion4 = iAFF()
        self.fusion3 = iAFF()
        self.fusion2 = iAFF()
        self.fusion1 = iAFF()
        # self.csc1 = CSC(set_channels)
        # self.csc2 = CSC(set_channels)
        # self.csc3 = CSC(set_channels)
        # self.csc4 = CSC(set_channels)
        # self.csfi1 = CSFI(set_channels, set_channels // 2)
        # self.csfi2 = CSFI(set_channels, set_channels // 2)
        # self.csfi3 = CSFI(set_channels, set_channels // 2)
        # self.csfi4 = CSFI(set_channels, set_channels // 2)

        self.sup4 = nn.Sequential(
            # nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.sup3 = nn.Sequential(
            # nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.sup2 = nn.Sequential(
            # nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.sup1 = nn.Sequential(
            # nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.sup1 = nn.Sequential(
            # nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.sup = nn.Sequential(
            # nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        self.fusion = nn.Sequential(nn.Conv2d(set_channels, set_channels, kernel_size=1),
                                    nn.Conv2d(set_channels, set_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(set_channels), nn.ReLU(inplace=True))

    def forward(self, fea, shape):
        fea_1, fea_2, fea_3, fea_4 = fea
        # sppf = self.sppf(fea_4)
        fea_1 = self.dfa1(self.cbam1(fea_1))
        fea_2 = self.dfa2(self.cbam2(fea_2))
        fea_3 = self.dfa3(self.cbam3(fea_3))
        fea_4 = self.dfa4(self.cbam4(fea_4))

        fea_1 = self.ice1(in1=fea_1, in2=fea_2)
        fea_2 = self.ice1(in1=fea_2, in2=fea_1, in3=fea_3)
        fea_3 = self.ice2(in1=fea_3, in2=fea_2, in3=fea_4)
        fea_4 = self.ice2(in1=fea_4, in2=fea_3)
        # 最后一个模块不可单独预测
        # fea_4 = self.fusion4(fea_4, sppf)
        fea_3 = self.fusion3(fea_3, F.interpolate(fea_4, scale_factor=2, mode='bilinear'))
        fea_2 = self.fusion2(fea_2, F.interpolate(fea_3, scale_factor=2, mode='bilinear'))
        fea_1 = self.fusion3(fea_1, F.interpolate(fea_2, scale_factor=2, mode='bilinear'))

        # fea_2 = F.interpolate(fea_2, size=fea_1.shape[2:])
        # fea_1 = self.fusion(fea_1 * fea_2) + fea_1
        out = self.fusion(fea_1 * F.interpolate(fea_2, size=fea_1.shape[2:])) + fea_1
        out = F.interpolate(self.sup(out), size=shape, mode="bilinear")
        out1 = F.interpolate(self.sup1(fea_1), size=shape, mode="bilinear")
        out2 = F.interpolate(self.sup2(fea_2), size=shape, mode="bilinear")
        out3 = F.interpolate(self.sup3(fea_3), size=shape, mode="bilinear")
        out4 = F.interpolate(self.sup4(fea_4), size=shape, mode="bilinear")

        return out, out1, out2, out3, out4


class TestMODEL(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, img_size=224):
        super(TestMODEL, self).__init__()
        self.img_size = [img_size, img_size]
        self.encoder = ResNet()
        self.decoder = Decoder()

    def forward(self, x):
        fea = self.encoder(x)
        out = self.decoder(fea, self.img_size)
        return out


if __name__ == '__main__':
    from thop import profile

    input = torch.rand(1, 3, 256, 256)
    model = TestMODEL(3, 1)
    model.eval()
    with torch.no_grad():
        ouput = model(input)
        for i in range(5):
            print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

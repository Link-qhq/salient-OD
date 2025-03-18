import torch
from torch import nn
from models.module.resnet import ResNet
# from models.module.ICONet import DFA,
import torch.nn.functional as F
from models.module.BIPGNet import SDU, BIG
from models.module.FP import iAFF
from models.module.SPP import SPPF_LSKA
from models.module.M3Net import MultilevelInteractionBlock
from models.module.upgrade import DFA


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        set_channels = 64
        self.dfa1 = DFA(256, set_channels)
        self.dfa2 = DFA(512, set_channels)
        self.dfa3 = DFA(1024, set_channels)
        self.dfa4 = DFA(2048, set_channels)

        self.mib1 = MultilevelInteractionBlock(set_channels, set_channels, set_channels, num_heads=1, mlp_ratio=3)
        self.mib2 = MultilevelInteractionBlock(set_channels, set_channels, set_channels, num_heads=2, mlp_ratio=3)
        self.mib3 = MultilevelInteractionBlock(set_channels, set_channels, num_heads=4, mlp_ratio=3)

        self.sppf = SPPF_LSKA(2048, set_channels)
        self.fusion4 = iAFF()
        self.fusion3 = iAFF()
        self.fusion2 = iAFF()
        self.fusion1 = iAFF()
        self.sup4 = nn.Sequential(
            nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=1, stride=1, padding=0)
        )
        self.sup3 = nn.Sequential(
            nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=1, stride=1, padding=0)
        )
        self.sup2 = nn.Sequential(
            nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=1, stride=1, padding=0)
        )
        self.sup1 = nn.Sequential(
            nn.Conv2d(set_channels, set_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(set_channels, 1, kernel_size=1, stride=1, padding=0)
        )

        self.linear = nn.Conv2d(set_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, fea, shape):
        fea_1, fea_2, fea_3, fea_4 = fea
        sppf = self.sppf(fea_4)
        fea_1 = self.dfa1(fea_1)
        fea_2 = self.dfa2(fea_2)
        fea_3 = self.dfa3(fea_3)
        fea_4 = self.dfa4(fea_4)
        B, C, _h, _w = fea_1.shape  # 256 * 56 * 56
        fea_3 = fea_3.reshape(B, C, -1).transpose(1, 2)
        fea_4 = fea_4.reshape(B, C, -1).transpose(1, 2)
        fea_2 = fea_2.reshape(B, C, -1).transpose(1, 2)
        fea_1 = fea_1.reshape(B, C, -1).transpose(1, 2)

        fea_3 = self.mib3(fea_3, fea_4)
        fea_2 = self.mib2(fea_2, fea_3, fea_4)
        fea_1 = self.mib1(fea_1, fea_2, fea_3)
        fea_1 = fea_1.transpose(1, 2).reshape(B, C, _h, _w)
        fea_2 = fea_2.transpose(1, 2).reshape(B, C, _h // 2, _w // 2)
        fea_3 = fea_3.transpose(1, 2).reshape(B, C, _h // 4, _w // 4)
        fea_4 = fea_4.transpose(1, 2).reshape(B, C, _h // 8, _w // 8)

        out4 = self.fusion4(fea_4, sppf)
        out3 = self.fusion3(fea_3, F.interpolate(out4, scale_factor=2, mode="bilinear"))
        out2 = self.fusion2(fea_2, F.interpolate(out3, scale_factor=2, mode="bilinear"))
        out1 = self.fusion1(fea_1, F.interpolate(out2, scale_factor=2, mode="bilinear"))

        out = self.linear(out1)
        refine = torch.sigmoid(out)
        out = F.interpolate(out, size=shape, mode="bilinear")
        out1 = F.interpolate(self.sup1(out1 + F.interpolate(refine, size=out1.shape[2:])), size=shape, mode="bilinear")
        out2 = F.interpolate(self.sup2(out2 + F.interpolate(refine, size=out2.shape[2:])), size=shape, mode="bilinear")
        out3 = F.interpolate(self.sup3(out3 + F.interpolate(refine, size=out3.shape[2:])), size=shape, mode="bilinear")
        out4 = F.interpolate(self.sup4(out4 + F.interpolate(refine, size=out4.shape[2:])), size=shape, mode="bilinear")

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
    input = torch.rand(2, 3, 224, 224)
    model = TestMODEL(3, 1)
    ouput = model(input)
    for i in range(5):
        print(ouput[i].shape)
    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

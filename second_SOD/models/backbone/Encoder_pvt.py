import torch
import torch.nn as nn
from models.backbone.PVT_V2 import pvt_v2_b2


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = pvt_v2_b2()
        self.encoder.load_state_dict(
            torch.load('/home/amax/文档/qhq/second_SOD/pretrained_model/pvt_v2_b2.pth', weights_only=True), strict=False)
        print("\033[94mPre-trained PVTv2 weight loaded.\033[0m")

    def forward(self, x):
        out = self.encoder(x)
        return out
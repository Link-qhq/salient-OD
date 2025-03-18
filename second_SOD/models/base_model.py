import torch
from torch import nn
from models.backbone.Encoder_pvt import Encoder
from models.backbone.resnet import ResNet


class Net(nn.Module):
    """
        Detail-guided salient object detection network
        Progressively guided network of detailed information
    """

    def __init__(self, decoder, backbone='resnet', img_size=224, channel=64, salient_idx=0):
        super(Net, self).__init__()
        self.img_size = [img_size, img_size]
        self.salient_idx = salient_idx
        self.backbone = backbone
        self.encoder = Encoder() if backbone == 'pvt' else ResNet()  # pvt or resnet or vgg
        self.in_channel_list = [64, 128, 320, 512] if backbone == 'pvt' else [256, 512, 1024, 2048]
        self.decoder = decoder(in_channel_list=self.in_channel_list, channel=channel)

    def forward(self, x, shape=None):
        B, C, H, W = x.shape
        fea1, fea2, fea3, fea4 = self.encoder(x)
        if self.backbone == 'pvt':
            fea1 = fea1.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[0], H // 4, W // 4)
            fea2 = fea2.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[1], H // 8, W // 8)
            fea3 = fea3.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[2], H // 16, W // 16)
            fea4 = fea4.permute(0, 2, 1).contiguous().view(B, self.in_channel_list[3], H // 32, W // 32)

        out = self.decoder((fea1, fea2, fea3, fea4), x.shape[2:] if shape is None else shape)
        return out

    def cal_mae(self, pred, gt):
        total_mae, avg_mae, B = 0.0, 0.0, pred.shape[0]
        pred = torch.sigmoid(pred)
        with torch.no_grad():
            for b in range(B):
                total_mae += torch.abs(pred[b][0] - gt[b]).mean()
        return total_mae / B
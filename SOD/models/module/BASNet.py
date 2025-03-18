import torch
from torch import nn
from torch.nn import functional as F


class RefUnet(nn.Module):
    """
        U型结构， 进一步精炼预测图
    """
    def __init__(self, in_ch, inc_ch):
        super(RefUnet, self).__init__()

        self.conv0 = nn.Conv2d(in_ch, inc_ch, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(inc_ch, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        #####

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)

        #####

        self.conv_d4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.relu_d4 = nn.ReLU(inplace=True)

        self.conv_d3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.relu_d3 = nn.ReLU(inplace=True)

        self.conv_d2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.relu_d2 = nn.ReLU(inplace=True)

        self.conv_d1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.relu_d1 = nn.ReLU(inplace=True)

        self.conv_d0 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        hx = x
        hx = self.conv0(hx)

        hx1 = self.relu1(self.bn1(self.conv1(hx)))
        hx = self.pool1(hx1)

        hx2 = self.relu2(self.bn2(self.conv2(hx)))
        hx = self.pool2(hx2)

        hx3 = self.relu3(self.bn3(self.conv3(hx)))
        hx = self.pool3(hx3)

        hx4 = self.relu4(self.bn4(self.conv4(hx)))
        hx = self.pool4(hx4)

        hx5 = self.relu5(self.bn5(self.conv5(hx)))

        # hx = self.upscore2(hx5)
        hx = F.interpolate(hx5, scale_factor=2, mode="bilinear")

        d4 = self.relu_d4(self.bn_d4(self.conv_d4(torch.cat((hx, hx4), 1))))
        # hx = self.upscore2(d4)
        hx = F.interpolate(d4, scale_factor=2, mode="bilinear")

        d3 = self.relu_d3(self.bn_d3(self.conv_d3(torch.cat((hx, hx3), 1))))
        # hx = self.upscore2(d3)
        hx = F.interpolate(d3, scale_factor=2, mode="bilinear")

        d2 = self.relu_d2(self.bn_d2(self.conv_d2(torch.cat((hx, hx2), 1))))
        # hx = self.upscore2(d2)
        hx = F.interpolate(d2, scale_factor=2, mode="bilinear")

        d1 = self.relu_d1(self.bn_d1(self.conv_d1(torch.cat((hx, hx1), 1))))

        residual = self.conv_d0(d1)

        return x + residual

if __name__ == '__main__':
    input = torch.rand(1, 1, 224, 224)
    net = RefUnet(1, 64)
    ouput = net(input)
    print(ouput.shape)
import torch
from torch import nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable


def get_weight(mask, alpha=5, window_size=15, power=1):
    return torch.pow(1. + alpha * torch.abs(F.avg_pool2d(mask, kernel_size=window_size, stride=1, padding=window_size // 2) - mask), power)


def wiou_loss(pred, mask):
    weight = get_weight(mask, power=2)
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()


# L_IoU
def iou_loss(pred, mask):
    """
    :param pred: 预测值
    :param mask: 标签值
    :return: loss
    """
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# L_wBCE
def wbce(pred, mask):
    weit = get_weight(mask=mask, window_size=31)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wbce.mean()


def cel(pred, target):
    pred = pred.sigmoid()
    intersection = pred * target
    numerator = (pred - intersection).sum() + (target - intersection).sum()
    denominator = pred.sum() + target.sum()
    return numerator / (denominator + 1e-6)


# L_ssim
'''
每个像素点的产生的loss都与其附近的局部patch有关（这里是N*N的patch），
因此在训练的过程中，会对物体边缘部分的loss值加强，对非边缘部分抑制。
正式因为这个loss的存在，使得该算法可以关注到更多的目标显著性的边缘细节信息，
'''


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def rev_iou_loss(pred, mask):
    weight = get_weight(mask=mask, window_size=31)
    pred = torch.sigmoid(pred)
    pred = 1 - pred
    mask = 1 - mask
    rev_wiou = wiou_loss(pred, mask, weight)
    return rev_wiou


#   L = wbce + 0.5 × (wiou + woiou)


def bce_iou_ssim(pred, label, alpha=1.0, beta=3.0, theta=0.5):
    return alpha * F.binary_cross_entropy_with_logits(pred, label) + \
           beta * iou_loss(pred, label) + theta * (1.0 - ssim(pred, label))


def bce_iou(pred, mask, alpha=1.0, beta=1.0):
    return alpha * F.binary_cross_entropy_with_logits(pred, mask) + beta * iou_loss(pred, mask)


def bce_wiou(pred, mask, alpha=1.0, beta=1.0):
    return alpha * F.binary_cross_entropy_with_logits(pred, mask) + beta * wiou_loss(pred, mask)


def wbce_iou(pred, mask, alpha=1.0, beta=1.0):
    return alpha * wbce(pred, mask) + beta * iou_loss(pred, mask)


def wbce_wiou(pred, mask, alpha=1.0, beta=1.0):
    return alpha * wbce(pred, mask) + beta * wiou_loss(pred, mask)

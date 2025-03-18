import torch
from torch import nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable


def get_weight(mask, alpha=5., window_size=15):
    return 1 + alpha * torch.abs(F.avg_pool2d(mask, kernel_size=window_size, stride=1, padding=window_size // 2) - mask)


def wiou_loss(pred, mask, alpha=5., window_size=15):
    pred = torch.sigmoid(pred)
    weight = get_weight(mask, alpha=alpha, window_size=window_size)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return wiou.mean()


def dice_loss(pred, mask):
    inter = (pred * mask).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (pred.sum() + mask.sum() + eps)
    return 1.0 - dice


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
    weit = get_weight(mask, window_size=31)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# L_wBCE
def wbce(pred, mask, alpha=5, window_size=15):
    weit = get_weight(mask=mask, alpha=alpha, window_size=window_size)
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
                window = window.to(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.to(img1.get_device())
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


def bce_dice(pred, mask, alpha=1.0, beta=1.0):
    bce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    return alpha * bce + beta * dice_loss(pred, mask)


def BCEDiceLoss(inputs, targets):
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice


class DSBCEDiceLoss(nn.Module):
    def __init__(self):
        super(DSBCEDiceLoss, self).__init__()

    def forward(self, inputs, target, teacher=False):
        # pred1, pred2, pred3, pred4, pred5 = tuple(inputs)
        if isinstance(target, tuple):
            target = target[0]
        # target = target[:,0,:,:]
        loss1 = BCEDiceLoss(inputs[:, 0, :, :], target)
        loss2 = BCEDiceLoss(inputs[:, 1, :, :], target)
        loss3 = BCEDiceLoss(inputs[:, 2, :, :], target)
        loss4 = BCEDiceLoss(inputs[:, 3, :, :], target)
        loss5 = BCEDiceLoss(inputs[:, 4, :, :], target)

        return loss1 + loss2 + loss3 + loss4 + loss5


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, target):
        if isinstance(target, tuple):
            target = target[0]
        assert inputs.shape[1] == 5

        loss1 = F.binary_cross_entropy(inputs[:, 0, :, :], target)
        loss2 = F.binary_cross_entropy(inputs[:, 1, :, :], target)
        loss3 = F.binary_cross_entropy(inputs[:, 2, :, :], target)
        loss4 = F.binary_cross_entropy(inputs[:, 3, :, :], target)
        loss5 = F.binary_cross_entropy(inputs[:, 4, :, :], target)
        return loss1 + loss2 + loss3 + loss4 + loss5


class FLoss(nn.Module):
    def __init__(self, beta=0.3, log_like=False):
        super(FLoss, self).__init__()
        self.beta = beta
        self.log_like = log_like

    def _compute_loss(self, prediction, target):
        EPS = 1e-10
        N = prediction.size(0)
        TP = (prediction * target).view(N, -1).sum(dim=1)
        H = self.beta * target.view(N, -1).sum(dim=1) + prediction.view(N, -1).sum(dim=1)
        fmeasure = (1 + self.beta) * TP / (H + EPS)
        if self.log_like:
            loss = -torch.log(fmeasure)
        else:
            loss = 1 - fmeasure
        return loss.mean()

    def forward(self, inputs, target):
        loss1 = self._compute_loss(inputs[:, 0, :, :], target)
        loss2 = self._compute_loss(inputs[:, 1, :, :], target)
        loss3 = self._compute_loss(inputs[:, 2, :, :], target)
        loss4 = self._compute_loss(inputs[:, 3, :, :], target)
        loss5 = self._compute_loss(inputs[:, 4, :, :], target)
        return 1.0 * loss1 + 1.0 * loss2 + 1.0 * loss3 + 1.0 * loss4 + 1.0 * loss5


def edge_loss(pred, mask):
    laplace = torch.Tensor([[-1., -1., -1., ], [-1., 8., -1.], [-1., -1., -1.]]).view([1, 1, 3, 3])
    laplace = nn.Parameter(data=laplace, requires_grad=False).to(pred.device)
    pred = torch.abs(torch.tanh(F.conv2d(pred, laplace, padding=1)))
    mask = torch.abs(torch.tanh(F.conv2d(mask, laplace, padding=1)))
    return F.binary_cross_entropy_with_logits(pred, mask, reduction='none')



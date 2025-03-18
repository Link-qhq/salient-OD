import torch.optim
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim.lr_scheduler as sche
import numpy as np
from torch import nn


class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, minimum_lr=0, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration

        self.last_epoch = None
        self.base_lrs = []

        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return (base_lr - self.minimum_lr) * ((1 - (step / self.max_iteration)) ** self.gamma) + self.minimum_lr

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in
                   self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

        return lrs


def make_scheduler(opt,
                   total_num,
                   scheduler_type,
                   warmup_epoch,
                   lr_decay=0.9,
                   optim_scheduler='lambda',
                   step_size=30,
                   gamma=0.1):
    """
    :param opt:
    :param total_num:
    :param scheduler_type:
    :param warmup_epoch:
    :param lr_decay:
    :param optim_scheduler:
    :param step_size:
    :param gamma:
    :return:
    """
    def get_lr_coefficient(curr_epoch):
        nonlocal total_num
        # curr_epoch start from 0
        # total_num = iter_num if args["sche_usebatch"] else end_epoch
        if scheduler_type == "poly":
            coefficient = pow((1. - float(curr_epoch) / total_num), lr_decay)
        elif scheduler_type == "poly_warmup":
            turning_epoch = warmup_epoch
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1. / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                total_num -= turning_epoch - 1
                coefficient = pow((1 - float(curr_epoch) / total_num), lr_decay)
        elif scheduler_type == "cosine_warmup":
            turning_epoch = warmup_epoch
            if curr_epoch < turning_epoch:
                # 0,1,2,...,turning_epoch-1
                coefficient = 1. / turning_epoch * (1 + curr_epoch)
            else:
                # turning_epoch,...,end_epoch
                curr_epoch -= turning_epoch - 1
                total_num -= turning_epoch - 1
                coefficient = (1. + np.cos(np.pi * curr_epoch / total_num)) / 2
        elif scheduler_type == "linear":
            coefficient = 1. - abs((curr_epoch + 1) / (total_num + 1) * 2 - 1)
        elif scheduler_type == 'linear_warmup':
            if curr_epoch < warmup_epoch:
                coefficient = 1. - abs((curr_epoch + 1) / (warmup_epoch * 2 + 1) * 2 - 1)
            else:
                coefficient = 1. - abs(
                    (curr_epoch + 1 + total_num - 2 * warmup_epoch) / ((total_num - warmup_epoch) * 2 + 1) * 2 - 1)
        else:
            raise NotImplementedError
        return coefficient

    if optim_scheduler == 'stepLR':
        scheduler = sche.StepLR(opt, step_size=step_size, gamma=gamma)
    else:
        scheduler = sche.LambdaLR(opt, lr_lambda=get_lr_coefficient)
    return scheduler


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    b = torch.rand(1, 64, 8, 8)
    net = nn.Sequential(
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    epochs = 100
    sche = make_scheduler(opt, epochs, 'poly', 10, 0.9)
    li = []
    for epoch in range(epochs):
        for it in range(10):
            a = torch.rand(1, 64, 8, 8)
            out = net(a)
            loss = torch.abs(b - out).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        li.append(opt.param_groups[0]['lr'])
        sche.step()

    plt.plot(range(len(li)), li)
    plt.show()

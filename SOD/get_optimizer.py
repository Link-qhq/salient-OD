import torch
from torch import nn


def get_params(model):
    base, head = [], []
    for name, param in model.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            pass
        if "encoder" in name:
            base.append(param)
        else:
            head.append(param)
    return base, head


def get_adam(lr, model, coe=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    """ coe 主干网络训练系数 """
    base, head = get_params(model)
    return torch.optim.Adam(params=[
        {"params": base, "lr": lr * coe},
        {"params": head, "lr": lr}
    ], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def get_sgd(lr, model, coe=1.0, momentum=0.9, weight_decay=5e-4):
    base, head = get_params(model)
    return torch.optim.SGD(params=[
        {"params": base, "lr": lr * coe},
        {"params": head, "lr": lr}
    ], lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

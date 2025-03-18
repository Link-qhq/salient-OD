import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR


def get_params(model, freeze=True):
    """

    :param model:
    :param freeze: 是否冻结resnet的第一层
    :return:
    """
    base, head = [], []
    for name, param in model.named_parameters():
        if 'encoder.conv1' in name or 'encoder.bn1' in name:
            if freeze:
                pass
            else:
                base.append(param)
                continue
        if "encoder" in name:
            base.append(param)
        else:
            head.append(param)
    return base, head


def get_adam(lr, model, coe=1.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, freeze=False):
    """ coe 主干网络训练系数 """
    base, head = get_params(model, freeze=freeze)
    return torch.optim.Adam(params=[
        {"params": base, "lr": lr * coe},
        {"params": head, "lr": lr}
    ], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def get_sgd(lr, model, coe=1.0, momentum=0.9, weight_decay=5e-4, freeze=True):
    base, head = get_params(model, freeze=freeze)
    return torch.optim.SGD(params=[
        {"params": base, "lr": lr * coe},
        {"params": head, "lr": lr}
    ], lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)


def adjust_learning_rate(args, optimizer, epoch):
    def update_lr(lr):
        if args.lr_mode == 'step':
            return lr * (0.1 ** (epoch // args.step_loss))
        elif args.lr_mode == 'poly':
            # max_iter = max_batches * args.train_epochs
            return lr * (1 - epoch * 1.0 / args.train_epochs) ** 0.9
        else:
            raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    # if epoch == 0 and epoch < 10:  # warm up
    #     lr = args.lr * 0.99 * (epoch + 1) / 10 + 0.01 * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = update_lr(param_group['lr'])


def get_scheduler(args, opt):
    def get_lr(warm_epoch, num_epoch):
        def lr_fun(cur_epoch):
            if cur_epoch < warm_epoch:
                return 1 - abs((cur_epoch + 1) / (warm_epoch * 2 + 1) * 2 - 1)
            else:
                return 1 - abs(
                    (cur_epoch + 1 + num_epoch - 2 * warm_epoch) / ((num_epoch - warm_epoch) * 2 + 1) * 2 - 1)

        return lr_fun

    return LambdaLR(optimizer=opt, lr_lambda=get_lr(args.warm_up, args.train_epochs), last_epoch=-1)


def get_lr(warm_epoch, num_epoch):
    def lr_fun(cur_epoch):
        if cur_epoch < warm_epoch:
            return 1 - abs((cur_epoch + 1) / (warm_epoch * 2 + 1) * 2 - 1)
        else:
            return 1 - abs(
                (cur_epoch + 1 + num_epoch - 2 * warm_epoch) / ((num_epoch - warm_epoch) * 2 + 1) * 2 - 1)

    return lr_fun


def update_lr(args, opt, epoch):
    lr1, lr2 = opt.param_groups[0]['lr'], opt.param_groups[1]['lr']
    l1 = (1 - abs((epoch + 1) / (args.train_epochs + 1) * 2 - 1)) * lr1
    l2 = (1 - abs((epoch + 1) / (args.train_epochs + 1) * 2 - 1)) * lr2
    return l1, l2


def make_optimizer(model, args):
    if args.optimizer == 'sgd':
        return get_sgd(lr=args.lr, model=model, coe=args.coe, momentum=args.momen, weight_decay=args.decay, freeze=args.freeze)
    else:
        return get_adam(lr=args.lr, model=model, coe=args.coe, betas=args.betas, eps=args.eps,
                        weight_decay=args.decay, freeze=args.freeze)

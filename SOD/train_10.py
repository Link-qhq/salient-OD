import datetime
from torch.utils.tensorboard import SummaryWriter
from data.big_dataloader import RGBDataSet
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda import amp
from utils import save_hyperParams
from get_optimizer import *
from get_loss import *
from models.model_11 import TestMODEL
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import argparse
import numpy as np


def get_index(file):
    files = os.listdir(file)
    new_file = [int(f.split('_')[1]) for f in files if f.find('-') == -1]
    new_file.sort()
    return str(new_file[-1])


def compute_loss(out, label, loss_function="bce_iou", loss_weights=None, boundary=None, alpha=1.0, beta=1.0, theta=1.0):
    # loss_weights = [0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 多级监督损失占比
    if loss_weights is None:
        loss_weights = [1.0, 1.0, 1.0, 1.0]
    loss_fun = eval(loss_function)
    all_loss = 0.
    loss_list = []
    for i in range(len(loss_weights)):
        loss_list.append(loss_fun(out[i], label, alpha, beta) + loss_fun(out[i + 4], boundary, alpha, beta / 30.))
        all_loss += loss_weights[i] * loss_list[-1]
    loss_list.insert(0, all_loss)
    return loss_list
    # 总损失, ....


def fit(model, train_dl, test_dl, args, file_fold):
    if args.optimizer == 'sgd':
        opt = get_sgd(lr=args.lr, model=model, coe=args.coe, momentum=args.momen, weight_decay=args.decay)
    else:
        opt = get_adam(lr=args.lr, model=model, coe=args.coe, betas=args.betas, eps=args.eps, weight_decay=args.decay)
    epochs = args.train_epochs

    #
    # scheduler = CosineAnnealingLR(optimizer=opt, T_max=epochs, eta_min=0,
    #                               last_epoch=args.last_epoch - 1) if args.scheduler else None
    def get_lr(warm_epoch, num_epoch):
        def lr_fun(cur_epoch):
            if cur_epoch < warm_epoch:
                return 1 - abs((cur_epoch + 1) / (warm_epoch * 2 + 1) * 2 - 1)
            else:
                # return 0.5 * (1.5 + np.cos(np.pi * (cur_epoch - warm_epoch) / (num_epoch - warm_epoch)))
                return 1 - abs(
                    (cur_epoch + 1 + num_epoch - 2 * warm_epoch) / ((num_epoch - warm_epoch) * 2 + 1) * 2 - 1)
        return lr_fun

    scheduler = LambdaLR(optimizer=opt, lr_lambda=get_lr(args.warm_up, epochs), last_epoch=-1)
    scaler = amp.GradScaler() if args.amp else None
    # 保存当前训练的超参
    save_hyperParams(args, file_fold)
    sw = SummaryWriter(file_fold)
    global_step = 0
    model.train()
    for epoch in range(args.last_epoch, epochs):
        epoch_loss_out = 0.
        iters = 0
        # opt.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (epochs + 1) * 2 - 1)) * args.lr * 0.1
        # opt.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (epochs + 1) * 2 - 1)) * args.lr
        progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch + 1, epochs), ncols=140)
        opt.zero_grad()
        for i, data_batch in enumerate(progress_bar):
            iters = iters + 1
            images, label, boundary = data_batch[0].cuda(non_blocking=True).float(), \
                data_batch[1].cuda(non_blocking=True).float(), data_batch[2].cuda(
                non_blocking=True).float()
            if args.amp:  # 混合精度训练
                with amp.autocast():
                    # out4, out3, out2, out1, out4_1, out3_1, out2_1, out1_1
                    out = model(images)
                    loss = compute_loss(out, label, loss_function=args.loss_fun, boundary=boundary,
                                        loss_weights=args.weights, alpha=args.alpha, beta=args.beta, theta=args.theta)
                scaler.scale(loss[0]).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(images)
                loss = compute_loss(out, label, loss_function=args.loss_fun, boundary=boundary,
                                    loss_weights=args.weights, alpha=args.alpha, beta=args.beta, theta=args.theta)
                loss[0].backward()
                opt.step()
            opt.zero_grad()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            epoch_loss_out += loss[args.salient_idx + 1].cpu().data.item()
            progress_bar.set_postfix(loss=f'{(epoch_loss_out / iters):.5f}')
            # tensorboard 展示 lr loss
            global_step += 1
            sw.add_scalar('lr', opt.param_groups[0]['lr'], global_step=global_step)
            sw_loss = {'loss{}'.format(i): x for (i, x) in enumerate(loss) if i != 0}
            sw.add_scalars('loss', sw_loss, global_step=global_step)
        # Record
        fh = open(file_fold + args.record, 'a')
        if epoch == 0:
            fh.write('\n' + str(datetime.datetime.now()) + '\n')
            fh.write('Start record.\n')
            fh.write('Step: ' + str(epoch + 1) + ', current lr: ' + str(opt.param_groups[0]['lr']) + '\n')
        fh.write("{} epoch_loss: {:.8f}     \n".format(epoch + 1, epoch_loss_out / iters))
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(),
                       file_fold + str(epoch + 1) + '_' + str(epochs) + '.pth')  # gpu Tensor
        if epoch + 1 == epochs:
            fh.write(str(datetime.datetime.now()) + '\n')
            fh.write('End record.\n')
        fh.close()
        if args.scheduler:
            scheduler.step()


def training(args):
    nn.LeakyReLU()
    model = TestMODEL()  # 模型
    if args.pretrained:  # 接着上次训练
        model.load_state_dict(torch.load(args.pre_path))
        print("\033[94mPre-trained Model loaded from: \033[0m" + args.pre_path)
    else:  # 主干权重
        args.last_epoch = 0
        model.encoder.load_state_dict(torch.load(args.pretrained_model + 'resnet50.pth'), strict=False)
        print("\033[94mPre-trained ResNet weight loaded.\033[0m")

    train_dataset = RGBDataSet(root=args.data_root, sets=['DUTS-TR'], img_size=args.img_size, mode='train')
    train_dl = DataLoader(train_dataset, collate_fn=train_dataset.collate, batch_size=args.batch_size, shuffle=True,
                          pin_memory=False, num_workers=args.num_workers
                          )

    model.to(args.device)
    print("\033[91mStarting train.\033[0m")
    last_index = str(int(get_index(args.save_model)) + 1)
    file_fold = args.save_model + 'exp_' + last_index + '/'
    print("\033[91mSave fold: \033[0m" + file_fold)
    if not os.path.exists(file_fold):
        os.makedirs(file_fold)
    test_dl = None
    fit(model, train_dl, test_dl, args, file_fold=file_fold)

    torch.save(model.state_dict(), file_fold + args.method + '.pth')
    print('\033[91mSaved as \033[0m' + file_fold + args.method + '.pth.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--train', default=True, type=bool, help='train or not')
    parser.add_argument('--data_root', default='datasets', type=str, help='data path')
    parser.add_argument('--device', default='cuda:0', type=str, help="use gpu or cpu")
    parser.add_argument('--train_epochs', default=100, type=int, help='total training epochs')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--method', default='Model_12', type=str, help='M3Net with different backbone')
    parser.add_argument('--pretrained_model', default='pretrained_model/', type=str, help='load Pretrained model')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=12, type=int, help='dataloader num workers')
    parser.add_argument('--amp', default=True, type=bool, help="amp train")

    # lr optimizer
    parser.add_argument('--optimizer', default='sgd', type=str, help="adopt optimizer")
    parser.add_argument('--scheduler', default=True, type=bool, help="adopt scheduler")
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--warm_up', default=50, type=int, help='warm up epochs')
    parser.add_argument('--coe', default=0.1, type=float, help='learning coefficient')
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='beats coefficient')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps coefficient')
    parser.add_argument("--momen", type=float, default=0.9, help="optimizer coefficient")
    parser.add_argument("--decay", type=float, default=1e-4, help="optimizer coefficient")

    # loss function
    parser.add_argument('--loss_fun', type=str, default="bce_iou", help="adopt loss function")
    parser.add_argument('--alpha', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--beta', type=float, default=3.0, help="loss coefficient")
    parser.add_argument('--theta', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--weights', type=list, default=[1.0, 1.0, 1.0, 1.0], help="loss weight")

    # step train
    # parser.add_argument('--step1epochs', default=100, type=int, help='train epochs for the step 1')
    # parser.add_argument('--step2epochs', default=20, type=int, help='train epochs for the step 2')
    parser.add_argument('--step_epochs', default=[10, 20, 30, 40, 50], type=list, help='train epochs for the step 2')

    # save pre-trained
    parser.add_argument('--model_file', type=str, default='model_2', help="training model file")
    parser.add_argument('--salient_idx', default=3, type=int, help='salient map index')
    parser.add_argument('--save_model', default='results/', type=str, help='save model path')
    parser.add_argument('--record', default='record.txt', type=str, help='record file')
    parser.add_argument('--pretrained', default=False, type=bool, help='pre-trained')
    parser.add_argument('--pre_path', default='results/exp_42/55_100.pth', type=str, help='pre-trained path')
    parser.add_argument('--last_epoch', default=0, type=int, help='last saved epoch')
    args = parser.parse_args()
    training(args)

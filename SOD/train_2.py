import datetime
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# from data.dataloader import RGB_Dataset, Only_For_Test
from data.F3_dataloader import RGBDataSet
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from evaluation.measure import MAE, WeightedFmeasure, Smeasure, Emeasure
from torch.cuda import amp
from utils import save_hyperParams
from get_optimizer import *
from get_loss import *
from models.mode_2 import TestMODEL
from torch.optim.lr_scheduler import CosineAnnealingLR
from arguments import get_arguments
from data.ICON_dataloader import ICONData


def get_index(file):
    files = os.listdir(file)
    new_file = [int(f.split('_')[1]) for f in files]
    new_file.sort()
    return str(new_file[-1])


def evalute_step(metric, pred, gt):
    pred, gt = pred.cpu().detach().numpy(), gt.cpu().detach().numpy()
    metric["mae"].step(pred, gt)
    metric["SMeasure"].step(pred, gt)
    metric["wFMeasure"].step(pred, gt)
    metric["EMeasure"].step(pred, gt)


def compute_loss(out, label, loss_function="bce_iou", loss_weights=None, boundary=None, alpha=1.0, beta=1.0, theta=1.0):
    # loss_weights = [0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 多级监督损失占比
    if loss_weights is None:
        loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    loss_fun = eval(loss_function)
    all_loss = 0.
    loss_list = []
    for i in range(len(loss_weights)):
        loss_list.append(loss_fun(out[i], label, alpha, beta))
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
    scheduler = CosineAnnealingLR(optimizer=opt, T_max=epochs, eta_min=0,
                                  last_epoch=args.last_epoch - 1) if args.scheduler else None
    scaler = amp.GradScaler() if args.amp else None
    # 保存当前训练的超参
    save_hyperParams(args, file_fold)
    sw = SummaryWriter(file_fold)
    global_step = 0
    model.train()
    for epoch in range(args.last_epoch, epochs):
        epoch_loss_out = 0.
        iters = 0
        opt.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (epochs + 1) * 2 - 1)) * args.lr * 0.1
        opt.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (epochs + 1) * 2 - 1)) * args.lr
        progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch + 1, epochs), ncols=140)
        opt.zero_grad()
        for i, data_batch in enumerate(progress_bar):
            iters = iters + 1
            images, label = data_batch[0].cuda(non_blocking=True).float(), \
                data_batch[1].cuda(non_blocking=True).float()
            if args.amp:  # 混合精度训练
                with amp.autocast():
                    # pred1, pred2, pred3, pred4, pose
                    out = model(images)
                    loss = compute_loss(out, label, loss_function=args.loss_fun,
                                        loss_weights=args.weights, alpha=args.alpha, beta=args.beta, theta=args.theta)
                scaler.scale(loss[0]).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(images)
                loss = compute_loss(out, label, loss_function=args.loss_fun,
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
            sw_loss = {'loss{}'.format(i): x for (i, x) in enumerate(loss)}
            sw.add_scalars('loss', sw_loss, global_step=global_step)
        # Record
        fh = open(file_fold + args.record, 'a')
        if epoch == 0:
            fh.write('\n' + str(datetime.datetime.now()) + '\n')
            fh.write('Start record.\n')
            fh.write('Step: ' + str(epoch + 1) + ', current lr: ' + str(opt.param_groups[0]['lr']) + '\n')
        fh.write("{} epoch_loss: {:.8f}     \n".format(epoch + 1, epoch_loss_out / iters))
        # Test
        # if (epoch + 1) % fre_epoch == 0:
        #     test_loss, metric = test_one_epoch(epoch, epochs, model, test_dl)
        #     # 打印评价指标
        #     mae, sm, wfm = metric["mae"].get_result()["mae"], \
        #                    metric["SMeasure"].get_results()["sm"], \
        #                    metric["wFMeasure"].get_results()["wfm"]
        #     em = metric["EMeasure"].get_results()["em"]
        #     avgE, maxE, adpE = em["curve"].mean(), em["curve"].max(), em["adp"]
        #     metric_result = "DUTS-TE => loss: {:.8f}     mae: {:.5f}      " \
        #                     "avgE: {:.5f}     maxE: {:.5f}     adpE: {:.5f}     " \
        #                     "sm: {:.5f}     wfm: {:.5f}\n" \
        #         .format(test_loss, mae, avgE, maxE, adpE, sm, wfm)
        #     print(metric_result)
        #     fh.write(metric_result)
        # else:
        #     fh.write('\n')
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
    model = TestMODEL()  # 模型
    if args.pretrained:  # 接着上次训练
        model.load_state_dict(torch.load(args.pre_path))
        print("\033[94mPre-trained Model loaded from: \033[0m" + args.pre_path)
    else:  # 主干权重
        model.encoder.load_state_dict(torch.load(args.pretrained_model + 'resnet50.pth'), strict=False)
        print("\033[94mPre-trained ResNet weight loaded.\033[0m")

    train_dataset = RGBDataSet(root=args.data_root, sets=['DUTS-TR'], img_size=args.img_size, mode='train')
    train_dl = DataLoader(train_dataset, collate_fn=train_dataset.collate, batch_size=args.batch_size, shuffle=True,
                          pin_memory=False, num_workers=args.num_workers
                          )
    # train_dataset = ICONData(root=args.data_root, sets=['DUTS-TR'], mode='train')
    # train_dl = DataLoader(train_dataset, collate_fn=train_dataset.collate, batch_size=args.batch_size, shuffle=True,
    #                       pin_memory=False, num_workers=args.num_workers
    #                       )

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
    args = get_arguments()
    training(args)

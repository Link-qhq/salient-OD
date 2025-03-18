import datetime
import random
import cv2
import numpy as np
import torch
from models.base_model import Net
from data.Test_dataloader import TestDataSet
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.edge_dataloader import RGBDataSet
from evaluation.measure import MAE, Smeasure, Emeasure, WeightedFmeasure, Fmeasure
from get_scheduler import make_scheduler
from utils import save_hyperParams
from get_optimizer import *
from get_loss import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def fit(model, train_dl, args, file_fold, compute_loss):
    opt = make_optimizer(model=model, args=args)
    epochs = args.train_epochs
    scheduler = make_scheduler(opt=opt,
                               total_num=epochs,
                               scheduler_type=args.scheduler_type,
                               warmup_epoch=args.warm_up,
                               lr_decay=args.lr_decay,
                               optim_scheduler=args.optim_scheduler,
                               step_size=args.step_size,
                               gamma=args.gamma)
    # scaler = amp.GradScaler() if args.amp else None
    scaler = torch.GradScaler(args.device, enabled=True) if args.amp else None
    # 保存当前训练的超参
    save_hyperParams(args, file_fold)
    model.train()
    for epoch in range(args.last_epoch, epochs):
        epoch_loss_out = 0.
        iters = 1
        epoch_mae = 0
        progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch + 1, epochs), ncols=140)
        opt.zero_grad()
        for i, data_batch in enumerate(progress_bar):
            iters = iters + 1
            if args.use_edge:
                images, label, boundary = data_batch[0].to(args.device, non_blocking=True).float(), \
                    data_batch[1].to(args.device, non_blocking=True).float(), \
                    data_batch[2].to(args.device, non_blocking=True).float()
            else:
                images, label, boundary = data_batch[0].to(args.device, non_blocking=True).float(), \
                                          data_batch[1].to(args.device, non_blocking=True).float(), \
                                          None
            if args.amp:  # 混合精度训练
                with torch.autocast(args.device, enabled=True):
                    # with amp.autocast():
                    # 统一使用list返回 eg: [out] or [out, out1, out2 ...]
                    out = model(images)
                    loss = compute_loss(out, label,
                                        loss_function=args.loss_fun,
                                        boundary=boundary,
                                        loss_weights=args.weights,
                                        alpha=args.alpha,
                                        beta=args.beta,
                                        theta=args.theta)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(images)
                loss = compute_loss(out, label,
                                    loss_function=args.loss_fun,
                                    boundary=boundary,
                                    loss_weights=args.weights,
                                    alpha=args.alpha,
                                    beta=args.beta,
                                    theta=args.theta)
                loss.backward()
                opt.step()
            # 开启验证集计算mae
            if args.valid:
                epoch_mae += model.cal_mae(out[args.salient_idx], label).cpu().data.item()
            opt.zero_grad()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            epoch_loss_out += loss.cpu().data.item()
            # 可视化每轮平均损失和mae
            progress_bar.set_postfix(loss=f'{(epoch_loss_out / iters):.5f}', mae=f'{(epoch_mae / iters):.5f}')
        # Record
        fh = open(file_fold + args.record, 'a')
        if epoch == 0:
            fh.write('\n' + str(datetime.datetime.now()) + '\n')
            fh.write('Start record.\n')
            fh.write('Step: ' + str(epoch + 1) + ', current lr: ' + str(opt.param_groups[0]['lr']) + '\n')
        fh.write("{} epoch_loss: {:.8f}     epoch_mae: {:.8f}\n".format(epoch + 1, epoch_loss_out / iters, epoch_mae / iters))
        if (epoch + 1) in args.save_epoch:
            torch.save(model.state_dict(),
                       file_fold + str(epoch + 1) + '_' + str(epochs) + '.pth')  # gpu Tensor
        if epoch + 1 == epochs:
            fh.write(str(datetime.datetime.now()) + '\n')
            fh.write('End record.\n')
        fh.close()
        if args.scheduler:
            scheduler.step()
        # 用于断点续训
        torch.save(model.state_dict(),
                   file_fold + 'last.pth')  # gpu Tensor


def training(args, decoder, compute_loss):
    model = Net(decoder=decoder,
                backbone=args.backbone,
                img_size=args.img_size,
                channel=args.channel,
                salient_idx=args.salient_idx)  # 模型
    train_dataset = RGBDataSet(root=args.data_root,
                               sets=['RCDD'],  # 记得还原
                               img_size=args.img_size, mode='train',
                               edge_type=args.edge_type,
                               use_scale=args.use_scale,
                               use_edge=args.use_edge)
    train_dl = DataLoader(train_dataset,
                          collate_fn=train_dataset.collate,
                          batch_size=args.batch_size,
                          shuffle=True,
                          pin_memory=False,
                          num_workers=args.num_workers,
                          drop_last=True)
    # 保留结果文件夹
    last_index = args.method.split('_')[1]
    file_fold = args.save_model + 'exp_' + last_index + '/'
    print("\033[91mSave fold: \033[0m" + file_fold)
    if not os.path.exists(file_fold):
        os.makedirs(file_fold)
    # 断点续训
    if args.last_epoch != 0:
        model.load_state_dict(torch.load(file_fold + 'last.pth', weights_only=True))
    model.to(args.device)
    print("\033[91mStarting train.\033[0m")
    fit(model, train_dl, args, file_fold=file_fold, compute_loss=compute_loss)
    # 保存权重
    torch.save(model.state_dict(), file_fold + args.method + '.pth')
    print('\033[91mSaved as \033[0m' + file_fold + args.method + '.pth.')


def get_pred_dir(model, data_root='datasets/', save_path='preds/', img_size=352, device='cpu',
                 methods='DUT-O+DUTS-TE+ECSSD+HKU-IS+PASCAL-S+SOD', num_workers=0, idx=0):
    batch_size = 1
    test_paths = methods.split('+')
    for dataset_setname in test_paths:
        test_dataset = TestDataSet(data_root, [dataset_setname], img_size, 'test')
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers)

        progress_bar = tqdm(test_loader, desc=dataset_setname, ncols=140)
        with torch.no_grad():
            for i, data_batch in enumerate(progress_bar):
                images = data_batch[0]
                image_path = data_batch[3]
                images = images.to(device).float()
                outputs_saliency = model(images, shape=data_batch[2])[idx]
                pred = (torch.sigmoid(outputs_saliency)[0, 0] * 255).cpu().numpy()

                filename = image_path[0]
                # save saliency maps
                save_test_path = save_path + dataset_setname + '/'
                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path)
                cv2.imwrite(os.path.join(save_test_path, filename + '.png'), np.round(pred))


def test(args, decoder):
    save_path = args.save_model + 'exp_' + args.method.split('_')[1] + '/'
    print("\033[91mSave fold: \033[0m" + save_path)
    print('\033[91mStarting test.\033[0m')
    model = Net(decoder=decoder,
                backbone=args.backbone,
                img_size=args.img_size,
                channel=args.channel)  # 模型
    model.to(args.device)
    # model.load_state_dict(torch.load(save_path + args.method + '.pth', weights_only=True))
    model.load_state_dict(torch.load(save_path + 'Model_9_1.pth', weights_only=True))
    print('\033[91mLoaded from: ' + save_path + args.method + '.pth.\033[0m')
    model.eval()
    get_pred_dir(model,
                 data_root=args.data_root,
                 save_path=args.save_test,
                 img_size=args.img_size,
                 device=args.device,
                 methods=args.test_methods,
                 num_workers=args.num_workers,
                 idx=args.salient_idx)
    print('\033[91mPredictions are saved at ' + args.save_test + '.\033[0m')


def compute_metric(args):
    save_path = args.save_model + 'exp_' + args.method.split('_')[1] + '/'
    file_fold = save_path + "metric.txt"
    fh = open(file_fold, 'a')
    test_paths = args.test_methods.split('+')
    for dataset_name in test_paths:
        metric = {
            "mae": MAE(),
            "EMeasure": Emeasure(),
            "SMeasure": Smeasure(),
            "wFMeasure": WeightedFmeasure(),
            "FMeasure": Fmeasure()
        }
        gt_name_list, rs_dir_lists = [], []
        pred_dir = os.path.join(args.pred_root, dataset_name)
        gt_dir = os.path.join(args.gt_root, dataset_name)
        rs_dir_lists.append(pred_dir + '/')
        processbar = tqdm(os.listdir(pred_dir), desc="{}".format(dataset_name), ncols=140)
        for filename in processbar:
            gt_name_list.append(os.path.join(gt_dir, 'gt', filename))
            mask_path = os.path.join(gt_dir, 'gt', filename)
            pred_path = os.path.join(pred_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            metric["mae"].step(pred, mask)
            metric["SMeasure"].step(pred, mask)
            metric["wFMeasure"].step(pred, mask)
            metric["EMeasure"].step(pred, mask)
            metric["FMeasure"].step(pred, mask)
        mae, sm, wfm = metric["mae"].get_results()["mae"], \
            metric["SMeasure"].get_results()["sm"], \
            metric["wFMeasure"].get_results()["wfm"]
        em, fm = metric["EMeasure"].get_results()["em"], metric["FMeasure"].get_results()["fm"]
        avgE, maxE, adpE = em["curve"].mean(), em["curve"].max(), em["adp"]
        avgF, maxF, adpF = fm["curve"].mean(), fm["curve"].max(), fm["adp"]
        metric_result = "mae: {:.5f}      wfm: {:.5f}      sm: {:.5f}     " \
                        "avgE: {:.5f}     maxE: {:.5f}     adpE: {:.5f}     " \
                        "avgF: {:.5f}     maxF: {:.5f}     adpF: {:.5f}\n" \
            .format(mae, wfm, sm, avgE, maxE, adpE, avgF, maxF, adpF)
        print(metric_result)
        fh.write(dataset_name + "\n")
        fh.write(metric_result)
    fh.write('\n')
    fh.close()

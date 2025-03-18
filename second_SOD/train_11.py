import datetime
import cv2
import numpy as np
from data.edge_dataloader import RGBDataSet
from data.big_dataloader import TestDataSet
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
# from torch.cuda import amp
from utils import save_hyperParams
from get_optimizer import *
from get_loss import *
from models.model_11 import PGN
from torch.optim.lr_scheduler import LambdaLR, StepLR
import argparse
from evaluation.measure import MAE, Smeasure, Emeasure, WeightedFmeasure, Fmeasure


def compute_loss(out, label, loss_function="bce_iou", loss_weights=None, boundary=None, alpha=1.0, beta=1.0, theta=1.0):
    # loss_weights = [0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 多级监督损失占比
    if loss_weights is None:
        loss_weights = [1.0, 1.0, 1.0, 1.0]
    sal_loss, edge_loss = 0.0, 0.0
    for sal in out[0]:
        sal_loss += wiou_loss(sal, label, get_weight(label, alpha=5, window_size=3))
    for edge in out[1]:
        edge_loss += wbce(edge, boundary, window_size=3)
    fuse_loss = wbce(out[2], label, window_size=3)
    return sal_loss * loss_weights[0] + edge_loss * loss_weights[1] + fuse_loss * loss_weights[2]


def fit(model, train_dl, args, file_fold):
    # if args.optimizer == 'sgd':
    #     opt = get_sgd(lr=args.lr, model=model, coe=args.coe, momentum=args.momen, weight_decay=args.decay)
    # else:
    #     opt = get_adam(lr=args.lr, model=model, coe=args.coe, betas=args.betas, eps=args.eps, weight_decay=args.decay)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.decay)
    epochs = args.train_epochs

    def get_lr(warm_epoch, num_epoch):
        def lr_fun(cur_epoch):
            if cur_epoch < warm_epoch:
                return 1 - abs((cur_epoch + 1) / (warm_epoch * 2 + 1) * 2 - 1)
            else:
                return 1 - abs(
                    (cur_epoch + 1 + num_epoch - 2 * warm_epoch) / ((num_epoch - warm_epoch) * 2 + 1) * 2 - 1)

        return lr_fun

    # scheduler = LambdaLR(optimizer=opt, lr_lambda=get_lr(args.warm_up, epochs), last_epoch=-1)
    scheduler = StepLR(opt, step_size=30, gamma=0.1)
    # scaler = amp.GradScaler() if args.amp else None
    scaler = torch.GradScaler(args.device, enabled=True) if args.amp else None
    # 保存当前训练的超参
    save_hyperParams(args, file_fold)
    model.train()
    for epoch in range(args.last_epoch, epochs):
        epoch_loss_out = 0.
        iters = 1
        progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch + 1, epochs), ncols=140)
        opt.zero_grad()
        for i, data_batch in enumerate(progress_bar):
            iters = iters + 1
            images, label, boundary = data_batch[0].to(args.device, non_blocking=True).float(), \
                                      data_batch[1].to(args.device, non_blocking=True).float(), \
                                      data_batch[2].to(args.device, non_blocking=True).float()
            if args.amp:  # 混合精度训练
                with torch.autocast(args.device, enabled=True):
                    # with amp.autocast():
                    # [sal], [edge], pred
                    out = model(images)
                    loss = compute_loss(out, label, loss_function=args.loss_fun, boundary=boundary,
                                        loss_weights=args.weights, alpha=args.alpha, beta=args.beta, theta=args.theta)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(images)
                loss = compute_loss(out, label, loss_function=args.loss_fun, boundary=boundary,
                                    loss_weights=args.weights, alpha=args.alpha, beta=args.beta, theta=args.theta)
                loss.backward()
                opt.step()
            opt.zero_grad()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            epoch_loss_out += loss.cpu().data.item()
            progress_bar.set_postfix(loss=f'{(epoch_loss_out / iters):.5f}')
        # Record
        fh = open(file_fold + args.record, 'a')
        if epoch == 0:
            fh.write('\n' + str(datetime.datetime.now()) + '\n')
            fh.write('Start record.\n')
            fh.write('Step: ' + str(epoch + 1) + ', current lr: ' + str(opt.param_groups[0]['lr']) + '\n')
        fh.write("{} epoch_loss: {:.8f}     \n".format(epoch + 1, epoch_loss_out / iters))
        if (epoch + 1) in args.save_epoch:
            torch.save(model.state_dict(),
                       file_fold + str(epoch + 1) + '_' + str(epochs) + '.pth')  # gpu Tensor
        if epoch + 1 == epochs:
            fh.write(str(datetime.datetime.now()) + '\n')
            fh.write('End record.\n')
        fh.close()
        if args.scheduler:
            scheduler.step()


def training(args):
    model = PGN(backbone='pvt')  # 模型
    args.last_epoch = 0
    train_dataset = RGBDataSet(root=args.data_root, sets=['DUTS-TR'], img_size=args.img_size, mode='train')
    train_dl = DataLoader(train_dataset, collate_fn=train_dataset.collate, batch_size=args.batch_size, shuffle=True,
                          pin_memory=False, num_workers=args.num_workers, drop_last=True
                          )

    model.to(args.device)
    print("\033[91mStarting train.\033[0m")
    last_index = args.method.split('_')[1]
    file_fold = args.save_model + 'exp_' + last_index + '/'
    print("\033[91mSave fold: \033[0m" + file_fold)
    if not os.path.exists(file_fold):
        os.makedirs(file_fold)
    fit(model, train_dl, args, file_fold=file_fold)

    torch.save(model.state_dict(), file_fold + args.method + '.pth')
    print('\033[91mSaved as \033[0m' + file_fold + args.method + '.pth.')


def get_pred_dir(model, data_root='datasets/', save_path='preds/', img_size=352,
                 methods='DUT-O+DUTS-TE+ECSSD+HKU-IS+PASCAL-S+SOD', num_workers=0, idx=0):
    batch_size = 1
    test_paths = methods.split('+')
    for dataset_setname in test_paths:
        test_dataset = TestDataSet(data_root, [dataset_setname], img_size, 'test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

        progress_bar = tqdm(test_loader, desc=dataset_setname, ncols=140)
        with torch.no_grad():
            for i, data_batch in enumerate(progress_bar):
                images = data_batch[0]
                image_path = data_batch[3]
                images = images.to(args.device).float()
                outputs_saliency = model(images, shape=data_batch[2])[idx]
                pred = (torch.sigmoid(outputs_saliency)[0, 0] * 255).cpu().numpy()

                filename = image_path[0]
                # save saliency maps
                save_test_path = save_path + dataset_setname + '/'
                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path)
                cv2.imwrite(os.path.join(save_test_path, filename + '.png'), np.round(pred))


def test(args):
    save_path = args.save_model + 'exp_' + args.method.split('_')[1] + '/'
    print("\033[91mSave fold: \033[0m" + save_path)
    print('\033[91mStarting test.\033[0m')
    model = PGN(backbone='pvt', img_size=args.img_size)
    model.to(args.device)
    model.load_state_dict(torch.load(save_path + args.method + '.pth', weights_only=True))
    print('\033[91mLoaded from: ' + save_path + args.method + '.pth.\033[0m')
    model.eval()
    get_pred_dir(model, data_root=args.data_root, save_path=args.save_test, img_size=args.img_size,
                 methods=args.test_methods, num_workers=args.num_workers, idx=args.salient_idx)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--train', default=True, type=bool, help='train or not')
    parser.add_argument('--data_root', default='datasets', type=str, help='data path')
    parser.add_argument('--device', default='cuda:1', type=str, help="use gpu or cpu")
    parser.add_argument('--train_epochs', default=100, type=int, help='total training epochs')
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    parser.add_argument('--method', default='Model_11', type=str, help='M3Net with different backbone')
    parser.add_argument('--pretrained_model', default='pretrained_model/', type=str, help='load Pretrained model')
    parser.add_argument('--batch_size', default=24, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=12, type=int, help='dataloader num workers')
    parser.add_argument('--amp', default=True, type=bool, help="amp train")
    parser.add_argument('--salient_idx', default=2, type=int, help='salient map index')
    parser.add_argument('--save_epoch', default=[80, 85, 90, 95], type=list, help='salient map index')

    # test
    parser.add_argument('--test', default=True, type=bool, help='test or not')
    parser.add_argument('--save_test', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_methods', type=str, default='ECSSD+DUTS-TE+PASCAL-S+HKU-IS', help="test dataset list")

    # compute_metric
    parser.add_argument('--metric', default=True, type=bool, help='compute metric')
    parser.add_argument("--pred_root", type=str, default="preds", help="preds root")
    parser.add_argument("--gt_root", type=str, default="datasets", help="gt root")

    # lr optimizer
    parser.add_argument('--optimizer', default='adam', type=str, help="adopt optimizer")
    parser.add_argument('--scheduler', default=True, type=bool, help="adopt scheduler")
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--warm_up', default=5, type=int, help='warm up epochs')
    parser.add_argument('--coe', default=0.1, type=float, help='learning coefficient')
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='beats coefficient')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps coefficient')
    parser.add_argument("--momen", type=float, default=0.9, help="optimizer coefficient")
    parser.add_argument("--decay", type=float, default=0, help="optimizer coefficient")

    # loss function
    parser.add_argument('--loss_fun', type=str, default="bce_iou", help="adopt loss function")
    parser.add_argument('--alpha', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--beta', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--theta', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--weights', type=list, default=[1.0, 3.0, 3.0], help="loss weight")

    # save pre-trained
    parser.add_argument('--save_model', default='results/', type=str, help='save model path')
    parser.add_argument('--record', default='record.txt', type=str, help='record file')

    args = parser.parse_args()

    if args.train:
        training(args)
    if args.test:
        test(args)
    if args.metric:
        compute_metric(args)

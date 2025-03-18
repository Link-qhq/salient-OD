import time
from base_module import test, compute_metric, training, setup_seed
import os
from get_loss import *
from models.comparison.model_3 import Decoder
import argparse


# 消融实验: baseline + SeE block + MixAttention
def compute_loss(out, label, boundary=None, loss_function="bce_iou", loss_weights=None, alpha=1.0, beta=1.0, theta=1.0):
    if loss_weights is None:
        loss_weights = [1.0, 1.0, 1.0, 1.0]
    all_loss = 0.
    for i, x in enumerate(out):
        all_loss = all_loss + bce_iou(x, label, alpha, beta) * loss_weights[i]
    return all_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--train', default=True, type=bool, help='train or not')
    parser.add_argument('--data_root', default='../datasets', type=str, help='data path')
    parser.add_argument('--last_epoch', default=0, type=str, help='resume or new train')
    parser.add_argument('--use_edge', default=True, type=bool, help='is or not use edge supervision')
    parser.add_argument('--use_scale', default=True, type=bool, help='is or not use scale train')
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    parser.add_argument('--device', default='cuda:1', type=str, help="use gpu or cpu")
    parser.add_argument('--amp', default=True, type=bool, help="amp train")
    parser.add_argument('--train_epochs', default=100, type=int, help='total training epochs')

    # model
    parser.add_argument('--backbone', default='resnet', type=str, help='model backbone type')
    parser.add_argument('--channel', default=64, type=int, help='model backbone change channel')
    parser.add_argument('--salient_idx', default=0, type=int, help='salient map index')

    # datasets
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--edge_type', default='edges', type=str, choices=['edge', 'edges'], help='edge image type')
    parser.add_argument('--num_workers', default=12, type=int, help='dataloader num workers')

    # valid
    parser.add_argument('--valid', default=True, type=bool, help='open validate datasets')

    # test
    parser.add_argument('--test', default=True, type=bool, help='test or not')
    parser.add_argument('--save_test', default='../preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_methods', type=str, default='DUTS-TE', help="test dataset list")

    # compute_metric
    parser.add_argument('--metric', default=True, type=bool, help='compute metric')
    parser.add_argument("--pred_root", type=str, default="../preds", help="preds root")
    parser.add_argument("--gt_root", type=str, default="../datasets", help="gt root")

    # lr optimizer and scheduler
    parser.add_argument('--optimizer', default='sgd', type=str, help="adopt optimizer")
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--coe', default=0.1, type=float, help='learning coefficient')
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='beats coefficient')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps coefficient')
    parser.add_argument("--momen", type=float, default=0.9, help="optimizer coefficient")
    parser.add_argument("--decay", type=float, default=1e-4, help="optimizer coefficient")
    parser.add_argument("--freeze", type=bool, default=True, help="is or not freeze resnet first stage")

    # scheduler
    parser.add_argument('--scheduler', default=True, type=bool, help="adopt scheduler")
    parser.add_argument('--optim_scheduler', default='lambdaLR', type=str, choices=['lambdaLR', 'stepLR'],
                        help="optim scheduler")
    parser.add_argument('--scheduler_type', default='linear_warmup', type=str, help="scheduler type")
    parser.add_argument('--lr_decay', default=0.9, type=float, help="scheduler coefficient")
    parser.add_argument('--warm_up', default=5, type=int, help='warm up epochs')
    parser.add_argument('--gamma', default=0.1, type=int, help='stepLR coefficient')
    parser.add_argument('--step_size', default=30, type=int, help='step size of stepLR ')

    # loss function
    parser.add_argument('--loss_fun', type=str, default="bce_iou", help="adopt loss function")
    parser.add_argument('--alpha', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--beta', type=float, default=3.0, help="loss coefficient")
    parser.add_argument('--theta', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--weights', type=list, default=[1.0, 0.5, 0.25, 0.125], help="loss weight")

    # save pre-trained
    parser.add_argument('--save_model', default='results/comparision/', type=str, help='save model path')
    parser.add_argument('--record', default='record.txt', type=str, help='record file')
    parser.add_argument('--save_epoch', default=[80, 85, 90, 95], type=list, help='salient map index')
    parser.add_argument('--pretrained_model', default='pretrained_model/', type=str, help='load Pretrained model')

    # 根据train编号文件生成对应的exp文件
    args = parser.parse_args()
    file_name, extension = os.path.splitext(os.path.basename(__file__))
    args.method = file_name.replace('train', 'Model')
    if args.train_epochs == 64:
        args.save_epoch = [55, 58, 60]
    elif args.train_epochs == 100:
        args.save_epoch = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    elif args.train_epochs == 120:
        args.save_epoch = [100, 105, 110, 115]
    elif args.train_epochs == 200:
        args.save_epoch = [150, 180, 190, 195]
    setup_seed(42)
    stage_time = time.perf_counter()
    if args.train:
        training(args, decoder=Decoder, compute_loss=compute_loss)
        print("Total training time: {:.4f} hours".format((time.perf_counter() - stage_time) / 3600))
        stage_time = time.perf_counter()
    if args.test:
        test(args, decoder=Decoder)
        print("Total predict time: {:.4f} hours".format((time.perf_counter() - stage_time) / 3600))
        stage_time = time.perf_counter()
    if args.metric:
        compute_metric(args)
        print("Total metric time: {:.4f} hours".format((time.perf_counter() - stage_time) / 3600))

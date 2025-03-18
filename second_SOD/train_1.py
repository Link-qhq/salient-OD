from base_module import test, compute_metric, training
import os
from get_loss import *
from models.model_1 import Decoder


import argparse


def compute_loss(out, label, loss_function="bce_iou", loss_weights=None, boundary=None, alpha=1.0, beta=1.0, theta=1.0):
    # loss_weights = [0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 多级监督损失占比
    if loss_weights is None:
        loss_weights = [1.0, 1.0, 1.0, 1.0]
    sal_loss, edge_loss = 0.0, 0.0
    sal_loss += bce_iou(out[0], label, alpha, beta)
    edge_loss += bce_iou(out[1], boundary, alpha, beta / 30.0)
    return sal_loss + edge_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--train', default=False, type=bool, help='train or not')
    parser.add_argument('--data_root', default='datasets', type=str, help='data path')
    parser.add_argument('--last_epoch', default=0, type=str, help='resume or new train')
    parser.add_argument('--use_edge', default=False, type=bool, help='is or not use edge supervision')
    # model
    parser.add_argument('--device', default='cuda:1', type=str, help="use gpu or cpu")
    parser.add_argument('--train_epochs', default=64, type=int, help='total training epochs')
    parser.add_argument('--backbone', default='pvt', type=str, help='model backbone type')
    parser.add_argument('--channel', default=64, type=int, help='model backbone change channel')
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    # parser.add_argument('--method', default='Model_1', type=str, help='M3Net with different backbone')
    parser.add_argument('--pretrained_model', default='pretrained_model/', type=str, help='load Pretrained model')
    # datasets
    parser.add_argument('--batch_size', default=24, type=int, help='batch_size')
    parser.add_argument('--edge_type', default='edges', type=str, help='edge image type')
    parser.add_argument('--use_scale', default=True, type=bool, help='is or not use scale train')
    parser.add_argument('--num_workers', default=12, type=int, help='dataloader num workers')
    parser.add_argument('--amp', default=True, type=bool, help="amp train")
    parser.add_argument('--salient_idx', default=0, type=int, help='salient map index')
    parser.add_argument('--save_epoch', default=[80, 85, 90, 95], type=list, help='salient map index')

    # valid
    parser.add_argument('--valid', default=False, type=bool, help='open validate datasets')

    # test
    parser.add_argument('--test', default=True, type=bool, help='test or not')
    parser.add_argument('--save_test', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_methods', type=str, default='ECSSD+DUTS-TE+PASCAL-S+HKU-IS', help="test dataset list")

    # compute_metric
    parser.add_argument('--metric', default=True, type=bool, help='compute metric')
    parser.add_argument("--pred_root", type=str, default="preds", help="preds root")
    parser.add_argument("--gt_root", type=str, default="datasets", help="gt root")

    # lr optimizer and scheduler
    parser.add_argument('--optimizer', default='adam', type=str, help="adopt optimizer")
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--coe', default=0.1, type=float, help='learning coefficient')
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='beats coefficient')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps coefficient')
    parser.add_argument("--momen", type=float, default=0.9, help="optimizer coefficient")
    parser.add_argument("--decay", type=float, default=0, help="optimizer coefficient")
    parser.add_argument("--freeze", type=bool, default=True, help="is or not freeze resnet first stage")

    # scheduler
    parser.add_argument('--scheduler', default=True, type=bool, help="adopt scheduler")
    parser.add_argument('--optim_scheduler', default='stepLR', type=str, choices=['lambdaLR', 'stepLR'],
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
    parser.add_argument('--weights', type=list, default=[1.0, 3.0, 3.0], help="loss weight")

    # save pre-trained
    parser.add_argument('--save_model', default='results/', type=str, help='save model path')
    parser.add_argument('--record', default='record.txt', type=str, help='record file')

    args = parser.parse_args()
    file_name, extension = os.path.splitext(os.path.basename(__file__))
    args.method = file_name.replace('train', 'Model')
    if args.train:
        training(args, decoder=Decoder, compute_loss=compute_loss)
    if args.test:
        test(args, decoder=Decoder)
    if args.metric:
        compute_metric(args)

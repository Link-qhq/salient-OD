import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--train', default=True, type=bool, help='train or not')
    parser.add_argument('--data_root', default='datasets', type=str, help='data path')
    parser.add_argument('--device', default='cuda:0', type=str, help="use gpu or cpu")
    parser.add_argument('--train_epochs', default=100, type=int, help='total training epochs')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--method', default='Model_2', type=str, help='M3Net with different backbone')
    parser.add_argument('--pretrained_model', default='pretrained_model/', type=str, help='load Pretrained model')
    parser.add_argument('--batch_size', default=16, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=12, type=int, help='dataloader num workers')
    parser.add_argument('--amp', default=True, type=bool, help="amp train")

    # lr optimizer
    parser.add_argument('--optimizer', default='sgd', type=str, help="adopt optimizer")
    parser.add_argument('--scheduler', default=False, type=bool, help="adopt scheduler")
    parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
    parser.add_argument('--coe', default=0.1, type=float, help='learning coefficient')
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple, help='beats coefficient')
    parser.add_argument('--eps', default=1e-8, type=float, help='eps coefficient')
    parser.add_argument("--momen", type=float, default=0.9, help="optimizer coefficient")
    parser.add_argument("--decay", type=float, default=1e-4, help="optimizer coefficient")

    # loss function
    parser.add_argument('--loss_fun', type=str, default="wbce_iou", help="adopt loss function")
    parser.add_argument('--alpha', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--beta', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--theta', type=float, default=1.0, help="loss coefficient")
    parser.add_argument('--weights', type=list, default=[1.0, 1.0, 1.0, 1.0, 1.0], help="loss weight")

    # step train
    # parser.add_argument('--step1epochs', default=100, type=int, help='train epochs for the step 1')
    # parser.add_argument('--step2epochs', default=20, type=int, help='train epochs for the step 2')
    parser.add_argument('--step_epochs', default=[10, 20, 30, 40, 50], type=list, help='train epochs for the step 2')

    # save pre-trained
    parser.add_argument('--model_file', type=str, default='model_2', help="training model file")
    parser.add_argument('--salient_idx', default=0, type=int, help='salient map index')
    parser.add_argument('--save_model', default='results/', type=str, help='save model path')
    parser.add_argument('--record', default='record.txt', type=str, help='record file')
    parser.add_argument('--pretrained', default=True, type=bool, help='pre-trained')
    parser.add_argument('--pre_path', default='results/exp_26/20_100.pth', type=str, help='pre-trained path')
    parser.add_argument('--last_epoch', default=20, type=int, help='last saved epoch')

    # test
    parser.add_argument('--test', default=True, type=bool, help='test or not')
    parser.add_argument('--save_test', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_methods', type=str, default='ECSSD+PASCAL-S', help="test dataset list")

    args = parser.parse_args()
    return args

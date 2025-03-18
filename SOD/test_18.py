import os
import argparse
import torch
from torch.autograd import Variable
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
# from models.module.F3Net import F3Net
import numpy as np
from data.big_dataloader import RGBDataSet
from models.model_16 import PGN


def get_pred_dir(model, data_root='datasets/', save_path='preds/', img_size=352,
                 methods='DUT-O+DUTS-TE+ECSSD+HKU-IS+PASCAL-S+SOD', num_workers=0, idx=0):
    batch_size = 1
    test_paths = methods.split('+')
    for dataset_setname in test_paths:
        test_dataset = RGBDataSet(data_root, [dataset_setname], img_size, 'test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers)

        progress_bar = tqdm(test_loader, desc=dataset_setname, ncols=140)
        with torch.no_grad():
            for i, data_batch in enumerate(progress_bar):
                images = data_batch[0]
                image_path = data_batch[3]
                images = images.cuda().float()
                outputs_saliency = model(images, shape=data_batch[2])[idx]
                pred = (torch.sigmoid(outputs_saliency)[0, 0] * 255).cpu().numpy()

                filename = image_path[0]
                # save saliency maps
                save_test_path = save_path + dataset_setname + '/'
                if not os.path.exists(save_test_path):
                    os.makedirs(save_test_path)
                cv2.imwrite(os.path.join(save_test_path, filename + '.png'), np.round(pred))


def testing(args):
    print("\033[91mSave fold: \033[0m" + args.save_path)
    print('\033[91mStarting test.\033[0m')
    model = PGN(img_size=args.img_size)
    # model = F3Net()
    model.to(args.device)
    model.load_state_dict(torch.load(args.save_path + args.method + '.pth'))
    # model.load_state_dict(torch.load('results/exp_36/95_100.pth'))
    print('\033[91mLoaded from: ' + args.save_path + args.method + '.pth.\033[0m')
    model.eval()
    get_pred_dir(model, data_root=args.data_root, save_path=args.save_test, img_size=args.img_size,
                 methods=args.test_methods, num_workers=args.num_workers, idx=args.salient_idx)
    print('\033[91mPredictions are saved at ' + args.save_test + '.\033[0m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help="use gpu or cpu")
    parser.add_argument('--img_size', default=352, type=int, help='network input size')
    parser.add_argument('--salient_idx', default=3, type=int, help='salient map index')
    parser.add_argument('--data_root', default='datasets', type=str, help='data path')
    parser.add_argument('--method', default='Model_16', type=str, help='M3Net with different backbone')
    parser.add_argument('--save_path', default='results/exp_52/', type=str, help='save model path')
    parser.add_argument('--num_workers', default=12, type=int, help='dataloader num workers')
    parser.add_argument('--save_test', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_methods', type=str, default='ECSSD+DUTS-TE+DUT-O+HKU-IS+PASCAL-S+SOD', help="test dataset list")

    args = parser.parse_args()
    testing(args)
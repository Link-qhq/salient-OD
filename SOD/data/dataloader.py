import os
import cv2
import sys
import re
import numpy as np
import torch.cuda
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
from threading import Thread

# filepath = os.path.split(__file__)[0]
# repopath = os.path.split(filepath)[0]
# sys.path.append(repopath)

from data.custom_transforms import *

Image.MAX_IMAGE_PIXELS = None


# 数据集预处理方式
def get_transform(img_size=224, mode="train"):
    comp = []
    if mode == 'train':
        # Data enhancement applied
        comp.append(static_resize(size=[img_size, img_size]))
        comp.append(random_scale_crop(range=[0.75, 1.25]))
        comp.append(random_flip(lr=True, ud=False))
        comp.append(random_rotate(range=[-10, 10]))
        comp.append(random_image_enhance(methods=['contrast', 'sharpness', 'brightness']))
        comp.append(tonumpy())
        comp.append(normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        comp.append(totensor())
    else:
        comp.append(static_resize(size=[img_size, img_size]))
        comp.append(tonumpy())
        comp.append(normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        comp.append(totensor())
    return transforms.Compose(comp)


def get_boundary(x):
    return 4.0 * x * (1.0 - x)


class RGB_Dataset(Dataset):
    def __init__(self, root, sets, img_size, mode):
        """
        :param root: 数据集根目录
        :param sets: 数据集集合[DUTE-TR,SOD,ECSSD....]
        :param img_size: 训练图片大小
        :param mode: 数据集模式(train/test)
        默认样本
            文件夹 imgs
            标签   gt
        """
        self.images, self.gts = [], []

        for set in sets:
            image_root, gt_root = os.path.join(root, set, 'imgs'), os.path.join(root, set, 'gt')

            images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                      f.lower().endswith(('.jpg', '.png'))]
            images = sort(images)

            gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png'))]
            gts = sort(gts)

            self.images.extend(images)
            self.gts.extend(gts)
        self.filter_files()

        self.size = len(self.images)
        self.transform = get_transform(img_size, mode)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')  # 样本直接转RGB
        gt = Image.open(self.gts[index]).convert('L')  # 标签直接转L灰度图,
        shape = gt.size  # 二维数组

        name = self.images[index].split(os.sep)[-1]
        name = os.path.splitext(name)[0]  # 图片名称
        sample = {'image': image, 'gt': gt, 'name': name, 'shape': shape}
        sample = self.transform(sample)  # 三维tensor
        # sample['gt_boundary'] = get_boundary(sample['gt'])
        return sample

    def filter_files(self):
        """
        对每个样本进行过滤，保证都是正确的样本
        :return:
        """
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size


def sort(x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(x, key=alphanum_key)


class Only_For_Test(Dataset):
    def __init__(self, root, sets, img_size):
        """
        :param root: 数据集根目录
        :param sets: 数据集集合[DUTE-TR,SOD,ECSSD....]
        :param img_size: 训练图片大小
        默认样本
            文件夹 imgs
            标签   gt
        """
        self.images, self.gts = [], []

        for set in sets:
            image_root, gt_root = os.path.join(root, set, 'imgs'), os.path.join(root, set, 'gt')

            images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                      f.lower().endswith(('.jpg', '.png'))]
            images = sort(images)

            gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png'))]
            gts = sort(gts)

            self.images.extend(images)
            self.gts.extend(gts)
        self.filter_files()

        self.size = len(self.images)
        self.transform_image = get_transform(img_size, mode='test')
        # self.transform_gt = transforms.Compose([
        #     tonumpy(),
        #     totensor()
        # ])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')  # 样本直接转RGB
        gt = Image.open(self.gts[index]).convert('L')  # 标签直接转L灰度图,
        shape = gt.size  # 二维数组 W * H
        name = self.images[index].split(os.sep)[-1]
        name = os.path.splitext(name)[0]  # 图片名称
        sample = {'image': image, 'shape': shape, 'name': name}
        sample = self.transform_image(sample)  # 三维tensor
        # sample['gt'] = self.transform_gt({'gt': gt})['gt']
        sample['gt'] = torch.from_numpy(np.array(gt, dtype=np.float32)).unsqueeze(0)  # [0-255]
        sample['gt_boundary'] = get_boundary(sample['gt'] / 255.)
        # sample['gt_boundary'] =
        return sample

    def filter_files(self):
        '''
        对每个样本进行过滤，保证都是正确的样本
        :return:
        '''
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size


if __name__ == '__main__':
    from arguments import get_arguments
    from matplotlib import pyplot as plt
    import cv2
    from PIL import Image

    args =get_arguments()
    dataset = RGB_Dataset(root=args.data_root, sets=['ECSSD'], img_size=args.img_size, mode='train')
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(dataloader):
        print("imgae: ", batch["image"].shape)
        print("gt: ", batch['gt'].shape)
    # print(dataset[0]['image'].shape)
    # GT = (dataset[0]['gt'].numpy() * 255).astype(np.uint8)
    # data = dataset[10]
    # gt = (data['gt_boundary'].numpy() * 255).astype(np.uint8)
    # # idx = gt == 255
    # # gt[idx] = 0
    # # gt = gt & (1 - gt)
    # # img = data["image"]
    # GT[GT == 255] = 0
    # img = Image.fromarray(GT[0], mode="L")
    # img.save("sdasd.png")
    # imgs = Image.fromarray(gt[0], mode="L")
    # imgs.save("imgs.png")
    # print(data['gt'].max())
    # print(data['gt_boundary'].max())


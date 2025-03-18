#!/usr/bin/python3
# coding=utf-8

import os, re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask /= 255
        return image, mask


class RandomCrop(object):
    def __call__(self, image, mask):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        else:
            return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask


########################### Dataset Class ###########################
class RGBDataSet(Dataset):
    def __init__(self, root, sets=["DUTS-TR"], img_size=352, mode='train'):
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        self.normalize = Normalize(mean=self.mean, std=self.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(img_size, img_size)
        self.totensor = ToTensor()
        self.images, self.gts = [], []
        self.mode = mode
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

    def __getitem__(self, idx):
        name = self.images[idx].split(os.sep)[-1]
        name = os.path.splitext(name)[0]  # 图片名称

        image = cv2.imread(self.images[idx])[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(self.gts[idx], 0).astype(np.float32)
        shape = mask.shape  # w, h

        if self.mode == 'train':
            image, mask = self.normalize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            return image, mask
        else:
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name

    def collate(self, batch):
        # size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        size = 352
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask

    def __len__(self):
        return len(self.images)

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


def sort(x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(x, key=alphanum_key)


########################### Testing Script ###########################


if __name__ == '__main__':
    dataset = RGBDataSet("../datasets", mode='train')
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os
import re
import cv2
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


class Normalize(object):
    def __init__(self):
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

    def __call__(self, image, mask=None, edge=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        if edge is None:
            return image, mask / 255
        return image, mask / 255, edge / 255


class RandomCrop(object):
    def __call__(self, image, mask=None, edge=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None, edge=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy(), edge[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, edge=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if edge is None:
            return image, mask
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, edge


class ToTensor(object):
    def __call__(self, image, mask=None, edge=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image

        mask = torch.from_numpy(mask)
        if edge is None:
            return image, mask

        edge = torch.from_numpy(edge)
        return image, mask, edge


class RGBDataSet(Dataset):
    def __init__(self, root, sets=["DUTS-TR"], img_size=352, mode="train"):
        assert os.path.exists(root), f"path '{root}' does not exist."
        self.normalize = Normalize()
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(img_size, img_size)
        self.totensor = ToTensor()
        self.mode = mode
        self.images, self.gts, self.boundaries = [], [], []
        for set in sets:
            image_root, gt_root, boundary_root = os.path.join(root, set, 'imgs'), os.path.join(root, set, 'gt'), \
                                            os.path.join(root, set, 'boundary')

            images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                      f.lower().endswith(('.jpg', '.png'))]
            images = sort(images)

            gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png'))]
            gts = sort(gts)

            boundary = [os.path.join(boundary_root, f) for f in os.listdir(boundary_root)
                        if f.lower().endswith(('.jpg', '.png'))]
            boundary = sort(boundary)

            self.images.extend(images)
            self.gts.extend(gts)
            self.boundaries.extend(boundary)

        self.filter_files()

        self.size = len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx].split(os.sep)[-1]
        name = os.path.splitext(name)[0]  # 图片名称

        image = cv2.imread(self.images[idx])[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(self.gts[idx], 0).astype(np.float32)
        boundary = cv2.imread(self.boundaries[idx], 0).astype(np.float32)
        shape = mask.shape  # w, h
        if self.mode == 'train':
            image, mask, boundary = self.normalize(image, mask, boundary)
            image, mask, boundary = self.randomcrop(image, mask, boundary)
            image, mask, boundary = self.randomflip(image, mask, boundary)
            return image, mask, boundary
        else:
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, shape, name

    def __len__(self):
        return len(self.images)

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask, boundaries = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            boundaries[i] = cv2.resize(boundaries[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        boundaries = torch.from_numpy(np.stack(boundaries, axis=0)).unsqueeze(1)
        return image, mask, boundaries

    def filter_files(self):
        """
        对每个样本进行过滤，保证都是正确的样本
        :return:
        """
        assert len(self.images) == len(self.gts)
        images, gts, boundaries = [], [], []
        for img_path, gt_path, b_path in zip(self.images, self.gts, self.boundaries):
            img, gt, bd = Image.open(img_path), Image.open(gt_path), Image.open(b_path)
            if img.size == gt.size and gt.size == bd.size:
                images.append(img_path)
                gts.append(gt_path)
                boundaries.append(b_path)
        self.images, self.gts, self.boundaries = images, gts, boundaries


def sort(x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(x, key=alphanum_key)

if __name__ == '__main__':
    dataset = RGBDataSet("../datasets", mode='train')
    print(dataset[0][2].shape)
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1)
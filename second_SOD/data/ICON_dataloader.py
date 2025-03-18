#!/usr/bin/python3
# coding=utf-8

import os
import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random


########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, body=None, detail=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        return image, mask / 255


class RandomCrop(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body = cv2.resize(body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        detail = cv2.resize(detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class RandomRotate(object):
    def rotate(self, x, random_angle, mode='image'):

        if mode == 'image':
            H, W, _ = x.shape
        else:
            H, W = x.shape

        random_angle %= 360
        image_change = cv2.getRotationMatrix2D((W / 2, H / 2), random_angle, 1)
        image_rotated = cv2.warpAffine(x, image_change, (W, H))

        angle_crop = random_angle % 180
        if random_angle > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180
        hw_ratio = float(H) / float(W)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)
        r = hw_ratio if H > W else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator

        w_crop = int(crop_mult * W)
        h_crop = int(crop_mult * H)
        x0 = int((W - w_crop) / 2)
        y0 = int((H - h_crop) / 2)
        crop_image = lambda img, x0, y0, W, H: img[y0:y0 + h_crop, x0:x0 + w_crop]
        output = crop_image(image_rotated, x0, y0, w_crop, h_crop)

        return output

    def __call__(self, image, mask=None, body=None, detail=None):

        do_seed = np.random.randint(0, 3)
        if do_seed != 2:
            if mask is None:
                return image
            return image, mask

        random_angle = np.random.randint(-10, 10)
        image = self.rotate(image, random_angle, 'image')

        if mask is None:
            return image
        mask = self.rotate(mask, random_angle, 'mask')

        return image, mask


class ColorEnhance(object):
    def __init__(self):

        # A:0.5~1.5, G: 5-15
        self.A = np.random.randint(7, 13, 1)[0] / 10
        self.G = np.random.randint(7, 13, 1)[0]

    def __call__(self, image, mask=None, body=None, detail=None):

        do_seed = np.random.randint(0, 3)
        if do_seed > 1:  # 1: # 1/3
            H, W, _ = image.shape
            dark_matrix = np.zeros([H, W, _], image.dtype)
            image = cv2.addWeighted(image, self.A, dark_matrix, 1 - self.A, self.G)
        else:
            pass

        if mask is None:
            return image
        return image, mask


class GaussNoise(object):
    def __init__(self):
        self.Mean = 0
        self.Var = 0.001

    def __call__(self, image, mask=None, body=None, detail=None):
        H, W, _ = image.shape
        do_seed = np.random.randint(0, 3)

        if do_seed == 0:  # 1: # 1/3
            factor = np.random.randint(0, 10)
            noise = np.random.normal(self.Mean, self.Var ** 0.5, image.shape) * factor
            noise = noise.astype(image.dtype)
            image = cv2.add(image, noise)
        else:
            pass

        if mask is None:
            return image
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask = torch.from_numpy(mask)
        return image, mask

    ########################### Config File ###########################


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class ICONData(Dataset):
    def __init__(self, root, sets, model_name="ICON-R", mode='train'):
        # self.cfg = cfg
        self.model_name = model_name
        self.mode = mode
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        self.normalize = Normalize(mean=self.mean, std=self.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(352, 352)
        # self.randomrotate = RandomRotate()
        # self.colorenhance = ColorEnhance()
        # self.gaussnoise = GaussNoise()
        self.totensor = ToTensor()
        # if cfg.mode == 'train':
        #     mode = '/Train/'
        # else:
        #     mode = '/Test/'
        # with open(cfg.datapath + mode + cfg.mode + '.txt', 'r') as lines:
        #     self.samples = []
        #     for line in lines:
        #         self.samples.append(line.strip())
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

    def __getitem__(self, idx):
        # name = self.images[idx].split('/')[-1]
        name = self.images[idx].split(os.sep)[-1]
        name = os.path.splitext(name)[0]  # 图片名称

        try:
            image = cv2.imread(self.images[idx])  # '.jpg')[:,:,::-1].astype(np.float32)
        except:
            print("######" + str(self.images[idx]))

        if self.mode == 'train':
            try:
                mask = cv2.imread(self.gts[idx], 0).astype(np.float32)
            except:
                print("#######" + str(self.gts[idx]))
            image, mask = self.normalize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomflip(image, mask)
            # image, mask = self.randomrotate(image, mask)
            # image, mask = self.colorenhance(image, mask)
            # image, mask = self.gaussnoise(image, mask)

            return image, mask
        else:
            shape = image.shape[:2]  # h, w, 3
            image = self.normalize(image)
            image = self.resize(image)
            image = self.totensor(image)

            return image, shape, name

    def __len__(self):
        return len(self.images)

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask


if __name__ == '__main__':
    dataset = ICONData(root='/root/autodl-tmp/salient_object_detection/datasets/datasets', sets=['DUTS-TR'], mode='train')
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, collate_fn=dataset.collate, batch_size=16,
                                                        num_workers=8)
    for i, batch in enumerate(dataloader):
        print(batch[0].shape, batch[1].shape)

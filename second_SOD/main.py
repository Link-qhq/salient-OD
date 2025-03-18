# import os
# import random
# import shutil
#
# import cv2
# import numpy as np
#
# def get_edge(root):
#     num = 0
#     for filename in os.listdir(os.path.join(root, 'gt')):
#         file_path = os.path.join(root, 'gt', filename)
#         image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#         # 高斯平滑（可选）以减少噪声
#         # blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
#         #
#         # 使用 Canny 边缘检测
#         # low_threshold = 128
#         # high_threshold = 300
#         # edges = cv2.Canny(image, low_threshold, high_threshold)
#         # print(os.path.join(root, 'edge', filename))
#         # cv2.imwrite(os.path.join(root, 'edge', filename), edges)
#
#         binary_mask = np.array(image, dtype='uint8')
#         # calculate the edge map with width 4, where (7+1)/2=4
#         contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         instance_contour = np.zeros(binary_mask.shape)
#         edge = cv2.drawContours(instance_contour, contours, -1, 255, 7)
#         edge = np.array(edge, dtype='uint8')
#         edge[binary_mask == 0] = 0
#         num += 1
#         print(filename)
#         cv2.imwrite(os.path.join(root, 'edges', filename), edge)
#     print(num)
# get_edge('/home/amax/文档/qhq/second_SOD/new_datasets/DUTS-TR')
# from PIL import Image
#
# # path = '/home/amax/文档/qhq/second_SOD/datasets'
# # new_path = '/home/amax/文档/qhq/second_SOD/new_datasets/DUTS-TR'
# # new_gt_root = os.path.join(new_path, 'gt')
# # new_imgs_root = os.path.join(new_path, 'imgs')
# # for img in os.listdir(new_imgs_root):
# #     if os.path.exists(os.path.join(new_gt_root, img)):
# #         os.remove()
# # print(len(os.listdir(new_imgs_root)))
# # print(len(os.listdir(new_gt_root)))
# # gt_root = os.path.join(path, 'HKU-IS', 'gt')
# # gt_list = os.listdir(gt_root)
# # random.shuffle(gt_list)
# # for i in range(100):
# #     a, b = os.path.join(gt_root, gt_list[i]), os.path.join(path, 'HKU-IS', 'imgs', gt_list[i])
# #     if os.path.exists(a) and os.path.exists(b):
# #         img, gt = Image.open(a), Image.open(b)
# #         if img.size == gt.size:
# #             shutil.copy(a, os.path.join(new_gt_root, 'hku' + gt_list[i]))
# #             shutil.copy(b,
# #                     os.path.join(new_imgs_root, 'hku' + gt_list[i]))
# # print(len(os.listdir(new_imgs_root)))
# # print(len(os.listdir(new_gt_root)))
# # dataset = ['DUTS-TR', 'DUTS-TE', 'ECSSD', 'PASCAL-S']
# # num = [10553, 100, 10, 10]
# # for i, data in enumerate(dataset):
# #     gt_root = os.path.join(path, data, 'gt')
# #     gt_list = os.listdir(gt_root)
# #     random.shuffle(gt_list)
# #     for i in range(num[i]):
# #         shutil.copy(os.path.join(gt_root, gt_list[i]), os.path.join(new_gt_root, gt_list[i]))
# #         shutil.copy(os.path.join(path, data, 'imgs', gt_list[i].replace('.png', '.png' if data == 'HKU-IS' else '.jpg')),
# #                     os.path.join(new_imgs_root, gt_list[i].replace('.png', '.png' if data == 'HKU-IS' else '.jpg')))
import torch
print(torch.cuda.is_available())

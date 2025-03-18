import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


def get_edge(root):
    i = 0
    for filename in os.listdir(os.path.join(root, 'gt')):
        i += 1
        if i > 1:
            break
        file_path = os.path.join(root, 'gt', filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # # 高斯平滑（可选）以减少噪声
        # blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
        #
        # # 使用 Canny 边缘检测
        low_threshold = 128
        high_threshold = 300
        edges = cv2.Canny(image, low_threshold, high_threshold)
        # binary_mask = np.array(image, dtype='uint8')
        # calculate the edge map with width 4, where (7+1)/2=4
        # contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # instance_contour = np.zeros(binary_mask.shape)
        # edge = cv2.drawContours(instance_contour, contours, -1, 255, 2)
        # edge = np.array(edge, dtype='uint8')
        # edge[binary_mask == 0] = 0
        # print(os.path.join(root, 'edge', filename))
        # cv2.imwrite(os.path.join(root, 'edge', filename), edges)
        # 显示结果
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(edges, cmap='gray')
        plt.title('Canny Edge Detection')
        plt.axis('off')

        plt.show()
get_edge('D:\\working\\salient_object_detection\\datasets\\datasets\\DUTS-TR')



# def test(epoch):
#     train_epochs = 100
#     warm_epoch = 0
#     lr = 0.01
#     # return lr * (1 - epoch * 1.0 / train_epochs) ** 0.5
#     return lr - lr * abs((epoch + 1 + train_epochs - 2 * warm_epoch) / ((train_epochs - warm_epoch) * 2 + 1) * 2 - 1)
# List = [test(i) for i in range(100)]
# plt.plot(range(100), List)
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# x = np.linspace(-10, 10, 100)  # 生成-10到10之间的100个点
# y = sigmoid(x)
# plt.plot(x, y, label='Sigmoid Function')
# plt.title("Sigmoid Function Curve")
# plt.xlabel("x")
# plt.ylabel("Sigmoid")
# plt.legend()
# plt.grid()
# plt.savefig('sigmoid.png', dpi=800)


# def relu(x):
#     return np.maximum(0, x)
# x = np.linspace(-10, 10, 100)
# y = relu(x)
# plt.plot(x, y, label="ReLu Function")
# plt.title("ReLU Function")
# plt.xlabel("x")
# plt.ylabel("ReLU")
# plt.legend()
# plt.grid()
# plt.savefig('relu.png', dpi=800)

# def leaky_relu(x):
#     alpha = 0.01
#     y = [i if i >= 0 else alpha * i for i in x]
#     return np.array(y)
# x = np.linspace(-10, 10, 100)
# y = leaky_relu(x)
# plt.plot(x, y, label="Leaky ReLu Function")
# plt.title("Leaky ReLU Function")
# plt.xlabel("x")
# plt.ylabel("Leaky ReLU")
# plt.legend()
# plt.grid()
# plt.savefig('Lrelu.png', dpi=800)

import os
import cv2
import numpy as np
import random

def get_edge(root):
    num = 0
    for filename in os.listdir(os.path.join(root, 'gt')):
        file_path = os.path.join(root, 'gt', filename)
        # file_path = os.path.join(root, 'imgs', filename)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # 高斯平滑（可选）以减少噪声
        blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
        #
        # # 使用 Canny 边缘检测
        low_threshold = 150
        high_threshold = 300
        edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
        # print(os.path.join(root, 'edge', filename))
        # cv2.imwrite(os.path.join(root, 'edge', filename), edges)

        # binary_mask = np.array(image, dtype='uint8')
        # calculate the edge map with width 4, where (7+1)/2=4
        # contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # instance_contour = np.zeros(binary_mask.shape)
        # edge = cv2.drawContours(instance_contour, contours, -1, 255, 4)
        # edge = np.array(edge, dtype='uint8')
        # edge[binary_mask == 0] = 0
        # num += 1
        print(filename)
        cv2.imwrite(os.path.join(root, 'edges', filename), edges)
    print(num)
get_edge('/root/autodl-tmp/SOD/datasets/DUTS-TE')
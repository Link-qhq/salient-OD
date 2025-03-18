import collections
# import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl
import torch
import torch.nn as nn
from matplotlib import colors
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib
# matplotlib.use('TkAgg')
from models.base_model import Net
# from models.model_16 import PGN
from models.model_9 import Decoder


def process_img(img_path):
    # 从测试集中读取一张图片，并显示出来
    # img_path = 'D:\\working\\salient_object_detection\\datasets\\datasets\\DUTS-TE\\imgs\\ILSVRC2012_test_00002863.jpg'
    img = Image.open(img_path)
    imgarray = np.array(img) / 255.0

    # plt.figure(figsize=(8, 8))
    # plt.imshow(imgarray)
    # plt.axis('off')
    # plt.show()
    # 将图片处理成模型可以预测的形式
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_img = transform(img).unsqueeze(0)
    return input_img


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def process_hook(hook_list):
    for hook_name in hook_list:
        module_name = model.decoder
        for sub_name in hook_name.split('.'):
            module_name = getattr(module_name, sub_name)
        # model.decoder[hook_name].register_forward_hook(get_activation(hook_name))
        module_name.register_forward_hook(get_activation(hook_name))
        # module_name.register_backward_hook(get_activation(hook_name))


# hook_names = ['sce1.conv_cross', 'sce1.conv']
# # 获取FIIM的结果，浅层特征
# conv_list = ['1']
# for conv in conv_list:
#     model.decoder.sce1.conv.register_forward_hook(get_activation(conv))  # 为sce3中的bn绑定钩子
# # model.decoder.sce3.register_forward_hook(get_activation())  # 为sce3中的bn绑定钩子
# _ = model(input_img)
# feature_map = activation['conv.1']  # 结果将保存在activation字典中


def show_feature(feature_map):
    # 计算所有通道的均值输出得到热力图
    heatmap = torch.mean(feature_map, dim=1).squeeze()
    # 使用Relu函数作用于热力图
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    fig = plt.figure(figsize=(30, 50))
    plt.imshow(heatmap)
    # plt.show()
    plt.savefig('../metric_img/feature/back.jpg', bbox_inches='tight', dpi=800)
    return heatmap
    # print(feature_map.shape)
    # feature_map = feature_map.squeeze(0)  # torch.Size([1, 64, 112, 112]) —> torch.Size([64, 112, 112])  去掉第0维 即batch_size维
    # gray_scale = torch.sum(feature_map, 0)
    # gray_scale = gray_scale / feature_map.shape[0]  # torch.Size([64, 112, 112]) —> torch.Size([112, 112])   从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
    # gray_scale = gray_scale.data.cpu().numpy()  # .data是读取Variable中的tensor  .cpu是把数据转移到cpu上  .numpy是把tensor转为numpy
    # fig = plt.figure(figsize=(30, 50))
    # plt.imshow(gray_scale)
    # plt.show()
    # return gray_scale


# show_feature(activation['1'])
# show_feature(activation['conv.1'])


def show_channels(fea, is_save=False, save_path=None):
    # 可视化结果，显示前64张
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(8, 8, 1)
    norm = colors.Normalize(vmin=0, vmax=1)
    im = plt.imshow(fea[0, 0, :, :], cmap='jet', norm=norm)
    plt.axis('off')
    for i in range(64):
        plt.subplot(8, 8, i + 1)
        plt.imshow(fea[0, i, :, :], cmap='jet')
        plt.axis('off')
    # 创建一个共享的颜色条
    cbar_ax = fig.add_axes([0.92, 0.12, 0.03, 0.75])  # 可调整颜色条的位置，这里示例在图形右侧
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=15)
    # plt.tight_layout(w_pad=0.1, h_pad=0.1)
    # plt.colorbar()
    plt.show()
    if is_save:
        plt.savefig(save_path, bbox_inches='tight', dpi=800)


# 图像类热力激活图
class GradCAM(nn.Module):
    def __init__(self):
        super(GradCAM, self).__init__()
        # 获取模型的特征提取层
        self.feature = nn.Sequential(collections.OrderedDict({
            name: layer for name, layer in model.named_children()
            if name not in ['avgpool', 'fc', 'softmax']
        }))
        # # 获取模型最后的平均池化层
        # self.avgpool = model.avgpool
        # # 获取模型的输出层
        # self.classifier = nn.Sequential(collections.OrderedDict([
        #     ('fc', model.fc),
        #     ('softmax', model.softmax)
        # ]))
        # 生成梯度占位符
        self.gradients = None

    # 获取梯度的钩子函数
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.feature(x)
        # 注册钩子
        h = x.register_hook(self.activations_hook)
        # 对卷积后的输出使用平均池化
        x = self.avgpool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # 获取梯度的方法
    def get_activations_gradient(self):
        return self.gradients

    # 获取卷积层输出的方法
    def get_activations(self, x):
        return self.feature(x)


# 获取热力图
def get_heatmap(model, img):
    model.eval()
    img_pre = model(img)
    # 获取预测最高的类别
    pre_class = torch.argmax(img_pre, dim=-1).item()
    # 获取相对于模型参数的输出梯度
    img_pre[:, pre_class].backward()
    # 获取模型的梯度
    gradients = model.get_activations_gradient()
    # 计算梯度相应通道的均值
    mean_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # 获取图像在相应卷积层输出的卷积特征
    activations = model.get_activations(input_img).detach()
    # 每个通道乘以相应的梯度均值
    for i in range(len(mean_gradients)):
        activations[:, i, :, :] *= mean_gradients[i]
    # 计算所有通道的均值输出得到热力图
    heatmap = torch.mean(activations, dim=1).squeeze()
    # 使用Relu函数作用于热力图
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()

    return heatmap


def process_heatmap(feature_map):
    # 计算所有通道的均值输出得到热力图
    heatmap = torch.mean(feature_map, dim=1).squeeze()
    # 使用Relu函数作用于热力图
    heatmap = F.relu(heatmap)
    # 对热力图进行标准化
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.numpy()
    # ax = sns.heatmap(heatmap, cmap='jet')
    # plt.show()
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # fig = plt.figure(figsize=(8, 8))
    # plt.tight_layout()
    # # plt.colorbar()
    # plt.imshow(heatmap, cmap='jet')
    # plt.axis('off')
    # plt.show()
    # plt.savefig('../metric_img/feature/back.jpg', bbox_inches='tight', dpi=800)
    return heatmap


# 合并热力图和原题，并显示结果
def merge_heatmap_image(heatmap, image_path):
    img = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    # # plt.imshow(heatmap)
    # # plt.show()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # grad_cam_img = cv2.addWeighted(img, 1.0, heatmap, 0.5, 0)
    grad_cam_img = heatmap * 0.5 + img * 1.0
    grad_cam_img = grad_cam_img / grad_cam_img.max()
    # grad_cam_img = grad_cam_img / 255.
    # 可视化图像
    # b, g, r = cv2.split(grad_cam_img)
    # grad_cam_img = cv2.merge([r, g, b])
    plt.figure(figsize=(8, 8))
    # plt.imshow(img)
    # pl.imshow(heatmap)
    plt.tight_layout()
    plt.imshow(grad_cam_img, cmap='jet')
    # plt.colorbar()
    plt.axis('off')
    # plt.show()


def draw(img_path, heap1, heap2):
    image = img_path
    gt_img = img_path.replace('.jpg', '.png').replace('imgs', 'gt')
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))

    axes[0, 0].imshow(cv2.cvtColor(cv2.resize(cv2.imread(image), (256, 256)), cv2.COLOR_BGR2RGB))
    axes[0, 0].axis('off')
    axes[0, 0].annotate('(a)', xy=(0.5, -0.10), xycoords='axes fraction',
                        ha='center', va='center', fontsize=15)
    axes[1, 0].imshow(cv2.cvtColor(cv2.resize(cv2.imread(gt_img), (256, 256)), cv2.COLOR_BGR2RGB))
    axes[1, 0].axis('off')
    axes[1, 0].annotate('(b)', xy=(0.5, -0.10), xycoords='axes fraction',
                        ha='center', va='center', fontsize=15)

    axes[0, 1].imshow(heap1, cmap='jet')
    axes[0, 1].axis('off')
    axes[0, 1].annotate('(c)', xy=(0.5, -0.10), xycoords='axes fraction',
                        ha='center', va='center', fontsize=15)
    # axes[0, 1].colorbar()

    axes[1, 1].imshow(heap2, cmap='jet')
    axes[1, 1].axis('off')
    axes[1, 1].annotate('(d)', xy=(0.5, -0.10), xycoords='axes fraction',
                        ha='center', va='center', fontsize=15)
    # axes[0, 1]._colorbars()
    plt.tight_layout(w_pad=0.1)
    plt.savefig('../metric_img/feature/all.png', bbox_inches='tight', dpi=800)

# def draw_channels(ch1, ch2):


if __name__ == '__main__':
    # 图片
    # img_path = 'D:\\working\\salient_object_detection\\datasets\\datasets\\DUTS-TE\\imgs\\ILSVRC2012_test_00002863.jpg'
    img_path = '/home/amax/文档/qhq/second_SOD/datasets/DUTS-TE/imgs/ILSVRC2012_test_00000003.jpg'
    # img_path = 'D:\\working\\salient_object_detection\\datasets\\datasets\\DUTS-TE\\imgs\\ILSVRC2012_test_00000003.jpg'
    input_img = process_img(img_path)
    # 模型
    # model = PGN()
    model = Net(decoder=Decoder,
                backbone='resnet',
                img_size=352,
                channel=64,
                salient_idx=0)  # 模型
    model.load_state_dict(torch.load('/home/amax/文档/qhq/second_SOD/results/exp_9/Model_9_1.pth', weights_only=True))
    for name, param in model.named_parameters():
        print(name)
    model.eval()

    # 定义钩子函数，获取指定层名称的特征
    activation = {}  # 保存获取的输出
    hook_names = ['se_att1.conv_out', 'se_att2.conv_out', 'se_att3.conv_out', 'se_att4.conv_out']
    process_hook(hook_names)
    out = model(input_img)

    # 特征通道可视化
    show_channels(activation[hook_names[0]])
    show_channels(activation[hook_names[1]])
    show_channels(activation[hook_names[2]])
    show_channels(activation[hook_names[3]])
    # # draw_channels(activation[hook_names[0]], activation[hook_names[1]])
    #
    # # 热力图
    # heatmap1 = process_heatmap(activation[hook_names[0]])
    # heatmap2 = process_heatmap(activation[hook_names[1]])
    #
    # draw(img_path, heatmap1, heatmap2)
    # # merge_heatmap_image(heatmap1, img_path)
    # # merge_heatmap_image(heatmap2, img_path)
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # 创建一个简单的二维数组作为虚拟图像
    # data = np.random.rand(256, 256)
    # plt.imshow(data, cmap='jet')
    # cbar = plt.colorbar()
    # plt.clf()  # 清除原始的虚拟图像，只保留Colorbar
    # plt.show()

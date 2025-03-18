import os
from compare import draw_compare
import cv2
import numpy as np
from matplotlib import pyplot as plt, colors
from tqdm import tqdm

from measure import Fmeasure


def plot_save_pr_curves(PRE, REC, method_names, lineSylClr, lineStyle, linewidth=2, xrange=(0.5, 1.0),
                        yrange=(0.5, 1.0),
                        dataset_name='TEST', save_dir='../metric_img/second_SOD/', save_fmt='png'):
    """
        绘制 PR 曲线
    :param PRE: 不同阈值下的precision => shape: [num_method_names, 255]
    :param REC: 不同阈值下的recall    => shape: [num_methods_names, 255]
    :param method_names: 模型方法名称[BASNet, F3Net, PoolNet, MINet-R, ...., Ours] => shape: num_method_names
    :param lineSylClr: 每种方法指定的曲线颜色                                        => shape: num_method_names
    :param linewidth: 每种方法指定的曲线现款                                         => shape: num_method_names
    :param xrange: X轴范围
    :param yrange: Y轴范围
    :param dataset_name: 数据集名称
    :param save_dir: 保存文件的目录
    :param save_fmt: 保存文件的格式
    :return:
    """
    # print(len(REC),len(PRE),len(lineSylClr),len(linewidth),len(method_names))
    print('\n')
    fig1 = plt.figure(1, figsize=(7, 5))
    num = PRE.shape[0]  # num_method_names
    for i in range(0, num):
        if len(np.array(PRE[i]).shape) != 0:
            plt.plot(REC[i], PRE[i], color=lineSylClr[i], linewidth=linewidth, label='F³Net' if method_names[i] == 'F3Net' else method_names[i],
                     linestyle=lineStyle[i])

    plt.xlim(xrange[0], xrange[1])  # 设置X轴范围
    plt.ylim(yrange[0], yrange[1])  # 设置Y轴范围
    xrange0 = xrange[0]
    yrange0 = yrange[0]
    #
    if xrange[0] * 10 - int(xrange[0] * 10) != 0:
        xrange0 = (int)(xrange[0] * 10 + 1) / 10
    if yrange[0] * 10 - (int)(yrange[0] * 10) != 0:
        yrange0 = (int)(yrange[0] * 10 + 1) / 10
    xyrange1 = np.arange(xrange0, xrange[1] + 0.01, 0.1)
    xyrange2 = np.arange(yrange0, yrange[1] + 0.01, 0.1)

    plt.tick_params(direction='in')  # 坐标轴刻度线方向为 内测

    yrange = list(yrange)
    plt.xticks(xyrange1, fontsize=16, fontname='serif')  # 设置X轴刻度值及其标签属性
    plt.yticks(xyrange2, fontsize=16, fontname='serif')

    ## draw dataset name
    plt.text((xrange[0] + xrange[1]) / 2.0, yrange[0] + (yrange[1] - yrange[0]) * 4 / 7, dataset_name,
             horizontalalignment='center', fontsize=28, fontname='serif', fontweight='bold')

    plt.xlabel('Recall', fontsize=24, fontname='serif')
    plt.ylabel('Precision', fontsize=24, fontname='serif')

    font1 = {'family': 'serif',
             'weight': 'normal',
             'size': 14,
             }

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [len(handles) - x for x in range(1, len(handles) + 1)]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left', prop=font1, ncol=2)
    plt.grid(linestyle='--')
    # plt.tight_layout()
    fig1.savefig(save_dir + dataset_name + "_pr_curves." + save_fmt, bbox_inches='tight', dpi=1000)
    print('>>PR-curves saved: %s' % (save_dir + dataset_name + "_pr_curves." + save_fmt))
    plt.cla()


def plot_save_fm_curves(FM, mybins, method_names, lineSylClr, lineStyle, linewidth=2, xrange=(0.5, 1.0), yrange=(0.5, 1.0),
                        dataset_name='TEST', save_dir='../metric_img/second_SOD/', save_fmt='png'):
    """
        绘制 Fmeasure 曲线
    :param FM: 不同阈值下的Fmeasure => shape: [num_method_name, 255]
    :param mybins: 阈值范围 [0, 256)
    :param method_names:
    :param lineSylClr:
    :param linewidth:
    :param xrange:
    :param yrange:
    :param dataset_name:
    :param save_dir:
    :param save_fmt:
    :return:
    """
    fig2 = plt.figure(2, figsize=(7, 5))
    num = FM.shape[0]
    for i in range(0, num):
        if (len(np.array(FM[i]).shape) != 0):
            plt.plot(np.array(mybins[0:-1]).astype(np.float64) / 255.0, FM[i], lineSylClr[i], linewidth=linewidth,
                     label='F³Net' if method_names[i] == 'F3Net' else method_names[i], linestyle=lineStyle[i])

    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])
    xrange0 = xrange[0]
    yrange0 = yrange[0]
    if xrange[0] * 10 - (int)(xrange[0] * 10) != 0:
        xrange0 = (int)(xrange[0] * 10 + 1) / 10
    if yrange[0] * 10 - (int)(yrange[0] * 10) != 0:
        yrange0 = (int)(yrange[0] * 10 + 1) / 10
    xyrange1 = np.arange(xrange0, xrange[1] + 0.01, 0.1)
    xyrange2 = np.arange(yrange0, yrange[1] + 0.01, 0.1)

    plt.tick_params(direction='in')
    plt.xticks(xyrange1, fontsize=16, fontname='serif')
    plt.yticks(xyrange2, fontsize=16, fontname='serif')

    ## draw dataset name
    plt.text((xrange[0] + xrange[1]) / 2.0, yrange[0] + (yrange[1] - yrange[0]) * 4 / 7, dataset_name,
             horizontalalignment='center', fontsize=28, fontname='serif', fontweight='bold')

    plt.xlabel('Thresholds', fontsize=24, fontname='serif')
    plt.ylabel('F-measure', fontsize=24, fontname='serif')

    font1 = {'family': 'serif',
             'weight': 'normal',
             'size': 14,
             }

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [len(handles) - x for x in range(1, len(handles) + 1)]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left', prop=font1, ncol=2)
    plt.grid(linestyle='--')
    # plt.tight_layout()
    fig2.savefig(save_dir + dataset_name + "_fm_curves." + save_fmt, bbox_inches='tight', dpi=1000)
    print('>>F-measure curves saved: %s' % (save_dir + dataset_name + "_fm_curves." + save_fmt))
    plt.cla()

if __name__ == '__main__':
    # draw_compare()
    com_len = 11
    all_dataset_precision, all_dataset_recall = np.empty((com_len, 256)), np.empty((com_len, 256))
    all_dataset_fm = np.empty((com_len, 256))
    all_methods = ['PiCANet', 'BASNet', 'CPD', 'F3Net', 'MINet', 'BPFINet', 'DCENet', 'RCSB', 'ICONet', 'EGOENet', 'Ours']
    # all_methods = ['Ours']
    all_dataset = ['DUTS-TE', 'ECSSD', 'HKU-IS', 'PASCAL-S']
    # all_dataset = ['ECSSD', 'HKU-IS', 'PASCAL-S']
    line_color = ['#7d44ee', '#d244ee', '#ee44b5', '#0b5af9', '#4460ee', '#ee4460', '#60ee44', '#8B6914', '#ee7d44', '#8595e4', '#f71308']
    line_style = ["-", "--", "-.", ":", "--", "-", "--", "--", "-.", ":", '-']

    root_dir = 'D:/working/SOD/comparison/SOD-SOTA-Saliency-maps'
    gt_dir = 'D:/working/salient_object_detection/datasets/datasets'
    for dataset_name in all_dataset:  # 数据集
        pre_dir = os.path.join(root_dir, dataset_name)
        for idx, method_name in enumerate(all_methods):  # 方法名
            method_dir = os.path.join(pre_dir, method_name)
            F_measure = Fmeasure()
            processbar = tqdm(os.listdir(method_dir), desc="{} / {}".format(dataset_name, method_name), ncols=140)
            for filename in processbar:
                if filename.split('.')[1] not in ['jpg', 'png']:
                    continue
                mask_path = os.path.join(gt_dir, dataset_name, 'gt', filename)
                pred_path = os.path.join(method_dir, filename)

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                # print(pred_path)
                # print(mask_path)
                if mask.size != pred.size:
                    pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]))
                F_measure.step(pred, mask)
            p, r = F_measure.get_results()['pr']['p'], F_measure.get_results()['pr']['r']
            all_dataset_precision[idx] = p
            all_dataset_recall[idx] = r
            all_dataset_fm[idx] = F_measure.get_results()['fm']['curve']
        #     print(p.shape)
        # print(all_dataset_precision.shape)
        plot_save_pr_curves(all_dataset_precision, all_dataset_recall, all_methods, line_color, line_style, dataset_name=dataset_name)
        plot_save_fm_curves(all_dataset_fm, np.linspace(0, 256, 257), all_methods, line_color, line_style, dataset_name=dataset_name)

    # draw_compare()
import torch
import numpy as np
from matplotlib import pyplot as plt


def plot_save_pr_curves(PRE, REC, method_names, lineSylClr, linewidth, xrange=(0.0, 1.0), yrange=(0.0, 1.0),
                        dataset_name='TEST', save_dir='../metric_imgs', save_fmt='png'):
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
    fig1 = plt.figure(1, figsize=(6, 7))
    num = PRE.shape[0]  # num_method_names
    for i in range(0, num):
        if len(np.array(PRE[i]).shape) != 0:
            plt.plot(REC[i], PRE[i], lineSylClr[i], linewidth=linewidth[i], label=method_names[i])

    plt.xlim(xrange[0], xrange[1])  # 设置X轴范围
    plt.ylim(yrange[0], yrange[1])  # 设置Y轴范围
    xrange0 = xrange[0]
    yrange0 = yrange[0]
    #
    if xrange[0] * 10 - int(xrange[0] * 10) != 0:
        xrange0 = (int)(xrange[0] * 10 + 1) / 10
    if yrange[0] * 10 - (int)(yrange[0] * 10) != 0:
        yrange0 = (int)(yrange[0] * 10 + 1) / 10
    xyrange1 = np.arange(xrange0, xrange[1] + 0.01, 0.2)
    xyrange2 = np.arange(yrange0, yrange[1] + 0.01, 0.05)

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
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center', prop=font1, ncol=2)
    plt.grid(linestyle='--')
    fig1.savefig(save_dir + dataset_name + "_pr_curves." + save_fmt, bbox_inches='tight', dpi=300)
    print('>>PR-curves saved: %s' % (save_dir + dataset_name + "_pr_curves." + save_fmt))


def plot_save_fm_curves(FM, mybins, method_names, lineSylClr, linewidth, xrange=(0.0, 1.0), yrange=(0.0, 1.0),
                        dataset_name='TEST', save_dir='./', save_fmt='pdf'):
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
    fig2 = plt.figure(2, figsize=(6, 7))
    num = FM.shape[0]
    for i in range(0, num):
        if (len(np.array(FM[i]).shape) != 0):
            plt.plot(np.array(mybins[0:-1]).astype(np.float) / 255.0, FM[i], lineSylClr[i], linewidth=linewidth[i],
                     label=method_names[i])

    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])
    xrange0 = xrange[0]
    yrange0 = yrange[0]
    if xrange[0] * 10 - (int)(xrange[0] * 10) != 0:
        xrange0 = (int)(xrange[0] * 10 + 1) / 10
    if yrange[0] * 10 - (int)(yrange[0] * 10) != 0:
        yrange0 = (int)(yrange[0] * 10 + 1) / 10
    xyrange1 = np.arange(xrange0, xrange[1] + 0.01, 0.2)
    xyrange2 = np.arange(yrange0, yrange[1] + 0.01, 0.05)

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
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center', prop=font1, ncol=2)
    plt.grid(linestyle='--')
    fig2.savefig(save_dir + dataset_name + "_fm_curves." + save_fmt, bbox_inches='tight', dpi=300)
    print('>>F-measure curves saved: %s' % (save_dir + dataset_name + "_fm_curves." + save_fmt))
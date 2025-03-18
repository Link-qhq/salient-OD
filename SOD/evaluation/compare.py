import os.path

from matplotlib import pyplot as plt
import cv2

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# all_methods = ['PiCANet', 'BASNet', 'CPD', 'F3Net', 'MINet', 'BPFINet', 'DCENet', 'RCSB', 'ICONet', 'Ours']
all_methods = ['PiCANet', 'BASNet', 'CPD', 'F3Net', 'MINet', 'BPFINet', 'DCENet', 'RCSB', 'ICONet', 'EGOENet', 'Ours']
all_methods.reverse()
all_dataset = ['DUTS-TE']

root_dir = 'D:/working/SOD/comparison/SOD-SOTA-Saliency-maps'
gt_dir = 'D:/working/salient_object_detection/datasets/datasets'
save_dir = 'D:/working/SOD/metric_img/module'


def get_compare():
    for idx1, dataset in enumerate(all_dataset):
        gt_path = os.path.join(gt_dir, dataset, 'gt')
        for id, filename in enumerate(os.listdir(gt_path)):
            if id < 4000:
                continue
            x, y = len(all_dataset), len(all_methods) + 1
            fig, axes = plt.subplots(1, len(all_methods) + 1, figsize=(20, 10))

            file_path = os.path.join(gt_path, filename)
            gt = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

            axes[0].imshow(gt)
            axes[0].axis('off')  # 关闭坐标轴
            for idx2, method in enumerate(all_methods):
                method_file_path = os.path.join(root_dir, dataset, method, filename)
                img = cv2.cvtColor(cv2.imread(method_file_path), cv2.COLOR_BGR2RGB)
                axes[idx2 + 1].imshow(img)
                axes[idx2 + 1].axis('off')  # 关闭坐标轴
            # 调整布局
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, dataset, filename), bbox_inches='tight', dpi=600)
            plt.cla()
            plt.close()


def draw_compare():
    # all_img = ['ILSVRC2012_test_00000498.png',
    #            'ILSVRC2012_test_00034815.png',
    #            'ILSVRC2012_test_00000003.png',
    #            'ILSVRC2012_test_00000259.png',
    #            'ILSVRC2012_test_00001361.png',
    #            # 'ILSVRC2012_test_00000481.png',
    #            'ILSVRC2012_test_00002863.png',
    #            'ILSVRC2012_test_00000128.png',
    #            'ILSVRC2012_test_00004423.png']
    all_img = ['ILSVRC2012_test_00001469.png',
               'ILSVRC2012_test_00003401.png',
               'ILSVRC2012_test_00006082.png',
               'ILSVRC2012_test_00004557.png',
               'sun_baslnamdoayixfvx.png']
    fig, axes = plt.subplots(len(all_img), len(all_methods) + 2, figsize=(14, 6))
    for idx1, img in enumerate(all_img):
        image = os.path.join(gt_dir, 'DUTS-TE', 'imgs', img.split('.')[0] + '.jpg')
        gt_img = os.path.join(gt_dir, 'DUTS-TE', 'gt', img)

        axes[idx1, 0].imshow(cv2.cvtColor(cv2.resize(cv2.imread(image), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[idx1, 0].axis('off')

        axes[idx1, 1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(gt_img), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[idx1, 1].axis('off')

        for idx2, method in enumerate(all_methods):
            pre_img = os.path.join(root_dir, 'DUTS-TE', method, img)
            axes[idx1, idx2 + 2].imshow(cv2.cvtColor(cv2.resize(cv2.imread(pre_img), (256, 256)), cv2.COLOR_BGR2RGB))
            axes[idx1, idx2 + 2].axis('off')

    # 添加行标签
    row_labels = ['Image', 'GT']
    for method in all_methods:
        row_labels.append(method)
    row_labels[8] = 'F³Net'
    for i, label in enumerate(row_labels):
        # fig.text(0.02, 0.8 - i * 0.4, label, ha='center', va='center', fontsize=12)
        axes[len(all_img) - 1, i].annotate(row_labels[i], xy=(0.5, -0.15), xycoords='axes fraction',
                                           ha='center', va='center', fontsize=12)
    # 添加列标签
    # col_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    col_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    for j, label in enumerate(col_labels):
        # fig.text(0.5 + j * 0.25, 0.03, label, ha='center', va='center', fontsize=12)
        axes[j, 0].annotate(col_labels[j], xy=(-0.2, 0.5), xycoords='axes fraction',
                            ha='center', va='center', fontsize=12)
    # plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])  # rect 用于避免标题重叠
    plt.tight_layout(w_pad=0.08, h_pad=0.01)
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 控制宽度和高度间距
    plt.savefig('../metric_img/dpi/second_SOD_compare.png', bbox_inches='tight', dpi=1000)


def draw_scale():
    imgs = ['ILSVRC2012_test_00000998.png',
            'ILSVRC2012_test_00093576.png',
            'sun_aaqmxnxoyktxhiwh.png',
            'ILSVRC2012_test_00000259.png']
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for idx1, img in enumerate(imgs):
        image = os.path.join(gt_dir, 'DUTS-TE', 'imgs', img.split('.')[0] + '.jpg')
        gt_img = os.path.join(gt_dir, 'DUTS-TE', 'gt', img)
        axes[0, idx1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(image), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[0, idx1].axis('off')

        axes[1, idx1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(gt_img), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[1, idx1].axis('off')

    # 添加列标签
    col_labels = ['Image', 'Mask']
    for j, label in enumerate(col_labels):
        axes[j, 0].annotate(col_labels[j], xy=(-0.2, 0.5), xycoords='axes fraction',
                            ha='center', va='center', fontsize=12)
    plt.tight_layout(w_pad=0.05, h_pad=0.05)

    plt.savefig('../metric_img/compare_edge.png', bbox_inches='tight', dpi=800)


def draw_show():
    all_img = ['ILSVRC2012_test_00000003.png',
               'ILSVRC2012_test_00002863.png',
               'ILSVRC2012_test_00004423.png']
    version = ['_1', '']
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for idx1, img in enumerate(all_img):
        image = os.path.join(gt_dir, 'DUTS-TE', 'imgs', img.split('.')[0] + '.jpg')
        gt_img = os.path.join(gt_dir, 'DUTS-TE', 'gt', img)

        axes[idx1, 0].imshow(cv2.cvtColor(cv2.resize(cv2.imread(image), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[idx1, 0].axis('off')

        axes[idx1, 1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(gt_img), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[idx1, 1].axis('off')

        for idx2, method in enumerate(version):
            pre_img = os.path.join('D:/working/SOD/preds', img.split('.')[0] + method + '.png')
            axes[idx1, idx2 + 2].imshow(cv2.cvtColor(cv2.resize(cv2.imread(pre_img), (256, 256)), cv2.COLOR_BGR2RGB))
            axes[idx1, idx2 + 2].axis('off')

    # 添加行标签
    row_labels = ['Image', 'GT', 'without EFPM', 'EFPM']
    # for method in all_methods:
    #     row_labels.append(method)
    for i, label in enumerate(row_labels):
        # fig.text(0.02, 0.8 - i * 0.4, label, ha='center', va='center', fontsize=12)
        axes[len(all_img) - 1, i].annotate(row_labels[i], xy=(0.5, -0.15), xycoords='axes fraction',
                                           ha='center', va='center', fontsize=12)
    # 添加列标签
    col_labels = ['(a)', '(b)', '(c)']
    for j, label in enumerate(col_labels):
        # fig.text(0.5 + j * 0.25, 0.03, label, ha='center', va='center', fontsize=12)
        axes[j, 0].annotate(col_labels[j], xy=(-0.1, 0.5), xycoords='axes fraction',
                            ha='center', va='center', fontsize=12)
    # plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])  # rect 用于避免标题重叠
    plt.tight_layout(w_pad=0.05, h_pad=0.2)
    # plt.subplots_adjust(wspace=0.3, hspace=0.3)  # 控制宽度和高度间距
    plt.savefig('../metric_img/compare_show.png', bbox_inches='tight', dpi=800)


def draw_layer():
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    filenames = os.listdir('../metric_img/layer')
    label = ['Image', 'Mask', '${P}$$^{4}_{b}$', '${P}$$^{3}_{b}$', '${P}$$^{2}_{b}$',
             '${P}$$^{1}_{b}$', '${P}$$^{4}_{g}$', '${P}$$^{3}_{g}$',
             '${P}$$^{2}_{g}$', '${P}$$^{1}_{g}$']
    for idx, filename in enumerate(filenames):
        filepath = os.path.join('../metric_img/layer', filename)
        axes[idx // 5, idx % 5].imshow(cv2.cvtColor(cv2.resize(cv2.imread(filepath), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[idx // 5, idx % 5].axis('off')
        axes[idx // 5, idx % 5].annotate(label[idx], xy=(0.5, -0.10), xycoords='axes fraction',
                                         ha='center', va='center', fontsize=12)
    plt.tight_layout(w_pad=0.1)
    plt.savefig('../metric_img/compare_layer.png', bbox_inches='tight', dpi=800)


def get_module():
    label = ['baseline', 'DFE', 'DFE+CFF', 'second', 'best', 'Mask']
    all_dir = [os.path.join('D:/working/SOD/metric_img/module', root) for root in label[0:3]]
    all_dir.append('D:/working/SOD/preds/DUTS-TE')
    all_dir.append('D:/working/SOD/comparison/SOD-SOTA-Saliency-maps/DUTS-TE/Ours')
    all_dir.append(os.path.join(gt_dir, 'DUTS-TE', 'gt'))
    for filename in os.listdir(all_dir[0]):
        fig, axes = plt.subplots(1, 6, figsize=(10, 2))
        for idx, root in enumerate(all_dir):
            axes[idx].imshow(
                cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(root, filename)), (256, 256)), cv2.COLOR_BGR2RGB))
            axes[idx].axis('off')
            axes[idx].annotate(label[idx], xy=(0.5, -0.10), xycoords='axes fraction',
                               ha='center', va='center', fontsize=12)
        plt.tight_layout(w_pad=0.1)
        plt.savefig('../metric_img/module_show/{}'.format(filename), bbox_inches='tight', dpi=800)
        plt.close()


def draw_module():
    imgs = ['ILSVRC2012_test_00001372.png', 'sun_bgkpkbwhcglpsher.png']
    labels = ['Image', 'Mask', 'baseline', '+DFEM', '+DEF,OIEM', '+DEF,OIE,EFEM']
    all_dir = [os.path.join('D:/working/SOD/metric_img/module', root) for root in
               os.listdir('D:/working/SOD/metric_img/module')]
    all_dir.insert(0, os.path.join(gt_dir, 'DUTS-TE', 'gt'))
    all_dir.insert(0, 'D:/working/salient_object_detection/datasets/datasets/DUTS-TE/imgs')
    all_dir.append('D:/working/SOD/comparison/SOD-SOTA-Saliency-maps/DUTS-TE/Ours')
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for id, filename in enumerate(imgs):
        for idx, root in enumerate(all_dir):
            if labels[idx] == 'Image':
                filename = filename.split('.')[0] + '.jpg'
            else:
                filename = filename.split('.')[0] + '.png'
            axes[id, idx].imshow(
                cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(root, filename)), (256, 256)), cv2.COLOR_BGR2RGB))
            axes[id, idx].axis('off')
            axes[id, idx].annotate(labels[idx], xy=(0.5, -0.10), xycoords='axes fraction',
                                   ha='center', va='center', fontsize=12)
    plt.tight_layout(w_pad=0.1)
    plt.savefig('../metric_img/compare_module.png', bbox_inches='tight', dpi=800)
    plt.close()


def draw_lose():
    # ILSVRC2012_test_00000003 ILSVRC2012_test_00000801.jpg ILSVRC2012_test_00053584.png sun_apcfycpnqlabadhz.png
    imgs = ['ILSVRC2012_test_00000003.png',
            'ILSVRC2012_test_00000801.png',
            'ILSVRC2012_test_00053584.png',
            'sun_apcfycpnqlabadhz.png']
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for idx1, img in enumerate(imgs):
        image = os.path.join(gt_dir, 'DUTS-TE', 'imgs', img.split('.')[0] + '.jpg')
        gt_img = os.path.join(gt_dir, 'DUTS-TE', 'gt', img)
        axes[0, idx1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(image), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[0, idx1].axis('off')

        axes[1, idx1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(gt_img), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[1, idx1].axis('off')

    # 添加列标签
    col_labels = ['Image', 'Mask']
    for j, label in enumerate(col_labels):
        axes[j, 0].annotate(col_labels[j], xy=(-0.2, 0.5), xycoords='axes fraction',
                            ha='center', va='center', fontsize=12)
    plt.tight_layout(w_pad=0.05, h_pad=0.05)

    plt.savefig('../metric_img/compare_object.png', bbox_inches='tight', dpi=800)


def draw_dataset(dataname, imgs):
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for idx1, img in enumerate(imgs):
        image = os.path.join(gt_dir, dataname, 'imgs', img.split('.')[0] + '.png')
        gt_img = os.path.join(gt_dir, dataname, 'gt', img)
        axes[0, idx1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(image), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[0, idx1].axis('off')

        axes[1, idx1].imshow(cv2.cvtColor(cv2.resize(cv2.imread(gt_img), (256, 256)), cv2.COLOR_BGR2RGB))
        axes[1, idx1].axis('off')

    # 添加列标签
    col_labels = ['Image', 'Mask']
    for j, label in enumerate(col_labels):
        axes[j, 0].annotate(col_labels[j], xy=(-0.2, 0.5), xycoords='axes fraction',
                            ha='center', va='center', fontsize=12)
    plt.tight_layout(w_pad=0.05, h_pad=0.05)

    plt.savefig('../metric_img/datasets/{}.png'.format(dataname), bbox_inches='tight', dpi=800)


if __name__ == '__main__':
    # get_compare()
    draw_compare()
    # draw_scale()
    # draw_show()
    # draw_layer()
    # get_module()
    # draw_module()
    # draw_lose()
    # draw_dataset('DUTS-TE', ['ILSVRC2012_test_00000023.png', 'ILSVRC2012_test_00000805.png', 'ILSVRC2012_test_00001862.png', 'ILSVRC2012_test_00004557.png'])
    # draw_dataset('ECSSD', ['0635.png', '0493.png', '0563.png', '0355.png'])
    # draw_dataset('PASCAL-S', ['162.png', '26.png', '197.png', '753.png'])
    # draw_dataset('HKU-IS', ['0209.png', '0364.png', '1093.png', '5246.png'])

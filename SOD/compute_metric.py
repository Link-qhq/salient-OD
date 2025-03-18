import cv2
import argparse
import os
from tqdm import tqdm
from evaluation.measure import MAE, Smeasure, Emeasure, WeightedFmeasure, Fmeasure


def start(args):
    # last_index = get_index(args.save_path)
    file_fold = args.save_path + "metric.txt"
    # if not os.path.exists(file_fold):
    #     os.makedirs(file_fold)
    fh = open(file_fold, 'a')
    test_paths = args.test_methods.split('+')
    for dataset_name in test_paths:
        metric = {
            "mae": MAE(),
            "EMeasure": Emeasure(),
            "SMeasure": Smeasure(),
            "wFMeasure": WeightedFmeasure(),
            "FMeasure": Fmeasure()
        }
        gt_name_list, rs_dir_lists = [], []
        pred_dir = os.path.join(args.pred_root, dataset_name)
        gt_dir = os.path.join(args.gt_root, dataset_name)
        rs_dir_lists.append(pred_dir + '/')
        processbar = tqdm(os.listdir(pred_dir), desc="{}".format(dataset_name), ncols=140)
        for filename in processbar:
            gt_name_list.append(os.path.join(gt_dir, 'gt', filename))
            mask_path = os.path.join(gt_dir, 'gt', filename)
            pred_path = os.path.join(pred_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            metric["mae"].step(pred, mask)
            metric["SMeasure"].step(pred, mask)
            metric["wFMeasure"].step(pred, mask)
            metric["EMeasure"].step(pred, mask)
            metric["FMeasure"].step(pred, mask)
        mae, sm, wfm = metric["mae"].get_results()["mae"], \
            metric["SMeasure"].get_results()["sm"], \
            metric["wFMeasure"].get_results()["wfm"]
        em, fm = metric["EMeasure"].get_results()["em"], metric["FMeasure"].get_results()["fm"]
        avgE, maxE, adpE = em["curve"].mean(), em["curve"].max(), em["adp"]
        avgF, maxF, adpF = fm["curve"].mean(), fm["curve"].max(), fm["adp"]
        metric_result = "mae: {:.5f}      sm: {:.5f}     wfm: {:.5f}     " \
                        "avgE: {:.5f}     maxE: {:.5f}     adpE: {:.5f}     " \
                        "avgF: {:.5f}     maxF: {:.5f}     adpF: {:.5f}\n" \
            .format(mae, sm, wfm, avgE, maxE, adpE, avgF, maxF, adpF)
        print(metric_result)
        fh.write(dataset_name + "\n")
        fh.write(metric_result)
    fh.write('\n')
    fh.close()
    # PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_name_list, rs_dir_lists)
    # lineSylClr = ['r-']  # curve style, same size with rs_dirs
    # linewidth = [2]  # line width, same size with rs_dirs
    # data_name = "ECSSD"
    # data_dir = "../preds/"
    # plot_save_fm_curves(FM,
    #                     mybins=np.arange(0, 256),
    #                     method_names=["M3Net"],
    #                     # method names, shape (num_rs_dir), will be included in the figure legend
    #                     lineSylClr=lineSylClr,  # curve styles, shape (num_rs_dir)
    #                     linewidth=linewidth,  # curve width, shape (num_rs_dir)
    #                     xrange=(0.0, 1.0),  # the showing range of x-axis
    #                     yrange=(0.0, 1.0),  # the showing range of y-axis
    #                     dataset_name=data_name,  # dataset name will be drawn on the bottom center position
    #                     save_dir=data_dir,  # figure save directory
    #                     save_fmt='png'
    #                     )
    # plot_save_pr_curves(PRE,  # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
    #                     REC,  # numpy array (num_rs_dir,255)
    #                     method_names=["M3Net"],
    #                     # method names, shape (num_rs_dir), will be included in the figure legend
    #                     lineSylClr=lineSylClr,  # curve styles, shape (num_rs_dir)
    #                     linewidth=linewidth,  # curve width, shape (num_rs_dir)
    #                     xrange=(0.5, 1.0),  # the showing range of x-axis
    #                     yrange=(0.5, 1.0),  # the showing range of y-axis
    #                     dataset_name=data_name,  # dataset name will be drawn on the bottom center position
    #                     save_dir=data_dir,  # figure save directory
    #                     save_fmt='png')


def other_methods(args):
    all_methods = ['PiCANet', 'BASNet', 'CPD', 'F3Net', 'MINet', 'DNA', 'BPFINet', 'EDNet', 'PDRNet', 'DCENet', 'RCSB', 'ICONet']
    file_fold = args.save_path + "resize_metric.txt"
    # if not os.path.exists(file_fold):
    #     os.makedirs(file_fold)
    fh = open(file_fold, 'a')
    test_paths = args.test_methods.split('+')
    for dataset_name in test_paths:
        fh.write(dataset_name + "\n")
        for method in all_methods:
            metric = {
                "mae": MAE(),
                "EMeasure": Emeasure(),
                "SMeasure": Smeasure(),
                "wFMeasure": WeightedFmeasure(),
                "FMeasure": Fmeasure()
            }
            pred_dir = os.path.join(args.pred_root, dataset_name, method)
            gt_dir = os.path.join(args.gt_root, dataset_name)
            processbar = tqdm(os.listdir(pred_dir), desc="{}".format(dataset_name), ncols=140)
            for filename in processbar:
                # print(filename)
                if filename.split('.')[1] not in ['jpg', 'png']:
                    continue
                mask_path = os.path.join(gt_dir, 'gt', filename)
                pred_path = os.path.join(pred_dir, filename)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                if mask.size != pred.size:
                    pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]))
                metric["mae"].step(pred, mask)
                metric["SMeasure"].step(pred, mask)
                metric["wFMeasure"].step(pred, mask)
                metric["EMeasure"].step(pred, mask)
                metric["FMeasure"].step(pred, mask)
            mae, sm, wfm = metric["mae"].get_results()["mae"], \
                           metric["SMeasure"].get_results()["sm"], \
                           metric["wFMeasure"].get_results()["wfm"]
            em, fm = metric["EMeasure"].get_results()["em"], metric["FMeasure"].get_results()["fm"]
            avgE, maxE, adpE = em["curve"].mean(), em["curve"].max(), em["adp"]
            avgF, maxF, adpF = fm["curve"].mean(), fm["curve"].max(), fm["adp"]
            metric_result = "mae: {:.5f}      sm: {:.5f}     wfm: {:.5f}     " \
                            "avgE: {:.5f}     maxE: {:.5f}     adpE: {:.5f}     " \
                            "avgF: {:.5f}     maxF: {:.5f}     adpF: {:.5f}\n"\
                .format(mae, sm, wfm, avgE, maxE, adpE, avgF, maxF, adpF)
            print(method)
            print(metric_result)
            fh.write(method + "\n")
            # fh.write(dataset_name + "\n")
            fh.write(metric_result)
            fh.write('\n')
    fh.write('\n')
    fh.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='metric_img/', type=str, help='save model path')
    parser.add_argument("--pred_root", type=str, default="D:/working/SOD/comparison/SOD-SOTA-Saliency-maps", help="preds root")
    parser.add_argument("--gt_root", type=str, default="D:/working/salient_object_detection/datasets/datasets", help="gt root")
    parser.add_argument('--test_methods', type=str, default='ECSSD+DUTS-TE+PASCAL-S+HKU-IS', help="test dataset list")
    args = parser.parse_args()
    # start(args)
    other_methods(args)
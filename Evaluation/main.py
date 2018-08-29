import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Iterator.GT_Pred_Iterator import GT_Pred_Iterator
from Experiment import Experiment
import os
from Plotter import Plotter
from JSONResults.results import *
from labels import *
from MetricsGeneratorPkg.Metric import Metric
from MetricsGeneratorPkg.MetricsGenerator import MetricsGenerator
from MetricsGeneratorPkg.MetricsGenerator import get_file_names
import sys
import numpy as np
import pandas as pd
import cv2
import glob

def temp():

    # Unique TrainIDs in gt: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 255}
    def get_file_names(dir_name):
        only_files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if
                      os.path.isfile(os.path.join(dir_name, f))]
        only_files.sort()
        return only_files

    server = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
    gta_trainIDs = server + "/home/mumu01/models/deeplab/datasets/cityscapes"

    groundTruthSearch = os.path.join(gta_trainIDs, "gtFine", "val", "*", "*_gtFine_labelTrainIds.png")
    groundTruthImgList = glob.glob(groundTruthSearch)
    groundTruthImgList.sort()

    # file_names = get_file_names(gta_trainIDs)
    file_names = groundTruthImgList

    empty_set_gt = set([])
    for idx, f_name in enumerate(file_names):
        print("\rImages Processed: {}".format(idx + 1), end=' ')

        sys.stdout.flush()

        img = cv2.imread(f_name, -1)
        empty_set_gt |= set(np.unique(img))

        idx += 1
        if idx % 100 == 0:
            print("\nSo far Unique TrainIDs in gt: {}".format((empty_set_gt)), end=' ')
            print()

    print("\nUnique TrainIDs in gt: {}".format((empty_set_gt)), end=' ')
def labelID_count():
    server = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
    gta_ids = server + "/home/mumu01/models/deeplab/datasets/gta/Ids"

    # gt_loc = dataset_loc + '/groundtruth'
    pred_loc = server + "/home/mumu01/exps" + '/exp54/vis_gta/raw_segmentation_results'

    # gt_paths = get_file_names(gt_loc)
    pred_paths = get_file_names(pred_loc)
    pred_paths_new = []
    for pred in pred_paths:
        pred_paths_new.append(os.path.basename(pred))

    gt_paths = [os.path.join(gta_ids, p) for p in pred_paths_new]
    # print(gt_paths)
    # exit()
    gt_pred_iter = GT_Pred_Iterator(ground_truth_filenames=gt_paths, prediction_filenames=pred_paths)

    empty_set_pred = set([])
    empty_set_gt = set([])
    i = 0
    for _, gt, _, pred in gt_pred_iter:
        empty_set_pred |= set(np.unique(pred))
        empty_set_gt |= set(np.unique(gt))
        print("\rImages Processed: {}".format(i + 1), end=' ')

        sys.stdout.flush()
        # print(set(np.unique(pred)))

        i += 1
        # if i>3:
        #     break
    print("\nUnique in gt: {}".format((empty_set_gt)), end=' ')
    print("\nUnique in pred: {}".format((empty_set_pred)), end=' ')
    print("\nTotal in gt: {}".format(len(empty_set_gt)), end=' ')
    print("\nTotal in pred: {}".format(len(empty_set_pred)), end=' ')


def usage_depth():
    from matplotlib.dates import date2num
    import cv2
    import random
    from PIL import Image

    def printer(arr):
        print('')
        print("min : ", np.amin(arr))
        print("max : ", np.amax(arr))
        print("unique : ", len(np.unique(arr)))
        print("unique with count : ", (np.unique(arr, return_counts=True)))

    def process_depthmap(depth, upper_th, lower_th):
        depth = np.array(depth, dtype=np.float64)

        invalid_indices = np.less_equal(depth, 0)

        disparity = (depth - 1) / 256.

        # scaling between 0 and 1.
        disparity *= 1.0 / disparity.max()

        mask = np.bitwise_and(np.greater_equal(upper_th, disparity),
                              np.greater(disparity, lower_th))

        mask[invalid_indices] = False

        return mask

    def get_th_imgs(gt, pred, mask):
        mask = np.argwhere(np.equal(mask, True))
        thresholded_gt = gt[mask[:, 0], mask[:, 1]]
        thresholded_pred = pred[mask[:, 0], mask[:, 1]]

        return thresholded_gt, thresholded_pred

    gt_loc = './datasets/sample/frankfurt_000000_000294_gtFine_labelIds.png'
    pred_loc = './datasets/sample/frankfurt_000000_000294_gtFine_labelIds.png'
    img_loc = './datasets/sample/frankfurt_000000_000294_leftImg8bit.png'
    depth_loc = './datasets/sample/frankfurt_000000_000294_disparity.png'

    gt = cv2.imread(gt_loc, -1)
    pred = cv2.imread(pred_loc, -1)
    img = cv2.imread(img_loc, -1)
    depthmap = cv2.imread(depth_loc, -1)

    mask = process_depthmap(depthmap, upper_th=1.0, lower_th=0.0)

    thresholded_gt, thresholded_pred = get_th_imgs(gt, pred, mask)
    print(gt.shape)
    print(np.shape(thresholded_gt))
    print(np.shape(thresholded_pred))
    exit()

    depthmap = translate(np.amin(depthmap), np.amax(depthmap), 0., 1., depthmap.copy())

    upper_distance_th = 1.0
    lower_distance_th = 0.9
    idx = np.argwhere((upper_distance_th >= depthmap) & (depthmap > lower_distance_th))

    thresholded_gt = gt[idx[:, 0], idx[:, 1]]
    thresholded_pred = pred[idx[:, 0], idx[:, 1]]

    labels_all = {label.name: label.id for label in labels if not label.id < 0}

    metric = Metric(len(labels_all), [l for l in labels_all], useUnlabeled=True)

    metric.update_matrix(thresholded_gt, thresholded_pred)

    _, _, _, _, conf_mat = metric.scores()

    confMatrix = np.asarray(conf_mat)

    Plotter.save_confusion_matrix_as_png(confMatrix, [l for l in labels_all], 'tmp.png',
                                         normalize=True, cmap=plt.cm.YlGn)


temp()
exit()
server = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
exps_root_dir = server + "/home/mumu01/exps"
# exps = ["53", "54"]
exps = ["55"]

for exp_num in exps:
    exp_dir = exps_root_dir + '/exp' + exp_num
    exp_obj = Experiment(exp_dir)
    m_g = MetricsGenerator(exp_obj, eval_dataset="cs")
    # m_g.complete_pipeline()
    # m_g.pipeline_3()
    # m_g.pipeline_4()
    m_g.pipeline_5()
# evaluator = CityscapesEvaluationWrapper(exp_obj, "cityscapes")
# if re_eval:

# print("Evaluating for cityscapes")
# evaluator.evaluate_using_CS_script()

# evaluator = CityscapesEvaluationWrapper(exp_obj, "gta")
# if re_eval:
#     print("Evaluating for GTA")
#     evaluator.evaluate_using_CS_script(is_gta=True)

# for exp_num in exps:
#     # exp_num = "53"
#     print("\n\t\tCalculating for exp: " + exp_num)
#     exp_dir = exps_root_dir + '/exp' + exp_num
#     exp_obj = Experiment(exp_dir)
#
#     for d in [[1.0, 0.75], [0.75, 0.5], [0.5, 0.25], [0.25, 0.0]]:
#         print("\n\t\t\tFor depth : ", d)
#         m_g = MetricsGenerator(exp_obj, eval_dataset="cs", depth_th=d)
#         m_g.complete_pipeline()
# m_g.plot_cm(True)

# cs_root_dir = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01/home/mumu01/models/deeplab/datasets/cityscapes"
# groundTruthSearch = os.path.join(cs_root_dir, "disparity", "val", "*", "*_disparity.png")
# groundTruthImgList = glob.glob(groundTruthSearch)
# groundTruthImgList.sort()
# # aachen_000000_000019_disparity.png
# print(groundTruthImgList[0])

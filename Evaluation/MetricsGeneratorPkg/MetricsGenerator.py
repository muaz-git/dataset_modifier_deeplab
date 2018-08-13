import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Evaluation.Iterator.GT_Pred_Iterator import GT_Pred_Iterator
from MetricsGeneratorPkg.Metric import Metric
import json
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
from Plotter import Plotter
from Evaluation.labels import labels



class MetricsGenerator(object):
    def __init__(self):
        pass

    @staticmethod
    def saveJSON(iterator: GT_Pred_Iterator, metric: Metric, json_path: str, ref_labels, row_idx):
        i = 0
        for _, gt, _, pred in iterator:
            print("\rImages Processed: {}".format(i + 1), end=' ')
            sys.stdout.flush()

            metric.update_matrix(gt, pred)

            i += 1

        _, _, _, _, conf_mat = metric.scores()
        metric.reset()

        # refining and getting rid of unwanted labels.
        conf_mat_summ = summaraize_cm(conf_mat, row_idx)
        ref_labels_summ = summarize_class_id_dict(ref_labels, row_idx)

        metric = Metric(len(ref_labels_summ), [l for l in ref_labels_summ], mat=conf_mat_summ, useUnlabeled=True)

        accuracy, avg_accuracy, IoU, mIoU, conf_mat = metric.scores()

        json_dict = {}
        json_dict["averageAccuracy"] = avg_accuracy.tolist()
        json_dict["averageScoreClasses"] = mIoU.tolist()
        accuracy = accuracy.tolist()
        #
        classScores_summ = {}

        for label, acc in zip(ref_labels_summ, accuracy):
            classScores_summ[label] = acc

        json_dict["classScores"] = classScores_summ
        json_dict["labels"] = ref_labels_summ
        json_dict["confMatrix"] = conf_mat.tolist()

        with open(json_path, 'w') as fp:
            json.dump(json_dict, fp, indent=4)


def get_file_names(dir_name):
    only_files = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
    only_files.sort()
    return only_files


def summaraize_cm(cm, indices_to_consider):
    return cm[indices_to_consider[:, None], indices_to_consider]


def summarize_class_id_dict(class_id_dict: dict, indices_to_consider):
    return {k: v for (k, v) in class_id_dict.items() if class_id_dict[k] in indices_to_consider}


if __name__ == '__main__':
    base_loc = '../datasets'

    dataset_loc = base_loc + '/cityscapes_54'

    gt_loc = dataset_loc + '/groundtruth'
    pred_loc = dataset_loc + '/predictions'

    gt_paths = get_file_names(gt_loc)
    pred_paths = get_file_names(pred_loc)

    gt_pred_iter = GT_Pred_Iterator(ground_truth_filenames=gt_paths, prediction_filenames=pred_paths)
    labels_all = {label.name: label.id for label in labels if not label.id < 0}
    metric = Metric(len(labels_all), [l for l in labels_all], useUnlabeled=True)

    row_idx = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])

    MetricsGenerator.saveJSON(iterator=gt_pred_iter, metric=metric, json_path="results_54_cs.json",
                              ref_labels=labels_all, row_idx=row_idx)
    exit()
    # i = 0
    # for gt_name, gt, pred_name, pred in gt_pred_iter:
    #     metric.update_matrix(gt, pred)
    #     print("\rImages Processed: {}".format(i + 1), end=' ')
    #     sys.stdout.flush()
    #
    #     i += 1

    # indices to consider

    # row_idx = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

    # ref_labels = [l for l in labels if not l.id < 0]

    with open('./results_53_cs.json') as f:
        data = json.load(f)

    confMatrix = np.asarray(data["confMatrix"])
    labels_all = {label.name: label.id for label in labels if not label.id < 0}

    confMatrix = summaraize_cm(confMatrix, row_idx)
    labels_all = summarize_class_id_dict(labels_all, row_idx)
    # print(np.shape(confMatrix))
    # print(len(labels_all))

    metric = Metric(len(labels_all), [l for l in labels_all], mat=confMatrix, useUnlabeled=True)
    Plotter.save_confusion_matrix_as_png(confMatrix, [l for l in labels_all], './exp_53_Sum.png',
                                         normalize=True, cmap=plt.cm.YlGn)
    print(metric.scores()[0], '\n')
    print(metric.scores()[1], '\n')  # 0.8220710291480284
    print(metric.scores()[2], '\n')
    print(metric.scores()[3])  # 0.5489342999658204

    # i=0
    # for gt_name, gt, pred_name, pred in gt_pred_iter:
    #     gt_name = os.path.basename(gt_name)
    #     pred_name = os.path.basename(pred_name)
    #     print("\rImages Processed: {}".format(i + 1), end=' ')
    #     if not gt_name[:-20] == pred_name[:-4]:
    #         raise ValueError('Gt name : ', gt_name[:-20], ' is not equal to Pred name : ', pred_name[:-4])
    #     sys.stdout.flush()
    #     i+=1
    # exit()
    # for l, idx in zip(ref_labels, range(len(ref_labels))):
    #     if not l.id == idx:
    #         raise ValueError("l.id : ", l.id, ' is not equal to idx : ', idx)
    #     print("using ", l.name)
    # for l in labels_name:
    #
    #     print(l)
    # metric = Metric(len(ref_labels), [l.name for l in ref_labels])
    #
    # MetricsGenerator.saveJSON(gt_pred_iter, metric, "hello.json", ref_labels)
    import collections
    # with open('/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01/home/mumu01/exps/exp53/custom_eval/resultPixelLevelSemanticLabeling_cityscapes.json') as f:
    #     data = json.load(f)

    # print(np.shape(data["confMatrix"]))
    # classScores = data['classScores']
    # label_dict = data['labels']
    # classScores = {k: classScores[k] for k in sorted(classScores)}
    # label_dict = {k: label_dict[k] for k in sorted(label_dict)}

    # data['classScores'] = classScores
    # data['labels'] = label_dict
    #
    # with open('./hello.json', 'w') as fp:
    #     json.dump(data, fp, indent=4)

    # classScores = {k:v for k,v in classScores.keys()}

    # i = 0
    # empty_set_pred = set([])
    # empty_set_gt = set([])
    # for _, gt, _, pred in gt_pred_iter:
    #     empty_set_pred |= set(np.unique(pred))
    #     empty_set_gt |= set(np.unique(gt))
    #     print("\rImages Processed: {}".format(i + 1), end=' ')
    #
    #     sys.stdout.flush()
    #     # print(set(np.unique(pred)))
    #
    #     i+=1
    #     # if i>3:
    #     #     break
    # print("\nUnique in gt: {}".format((empty_set_gt)), end=' ')
    # print("\nUnique in pred: {}".format((empty_set_pred)), end=' ')
    # print("\nTotal in gt: {}".format(len(empty_set_gt)), end=' ')
    # print("\nTotal in pred: {}".format(len(empty_set_pred)), end=' ')

    # cityscapes
    # Unique in gt: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    #                29, 30, 31, 32, 33}
    # Unique in pred: {7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33}
    # Total in gt: 33
    # Total in pred: 19

    # gta
    # Unique in gt: {0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33}
    # Unique in pred: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}
    # Total in gt: 20
    # Total in pred: 19

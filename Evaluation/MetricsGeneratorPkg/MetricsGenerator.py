import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Iterator.GT_Pred_Iterator import GT_Pred_Iterator
from MetricsGeneratorPkg.Metric import Metric
import json
import numpy as np
from os import listdir
from os.path import isfile, join
import sys
from Plotter import Plotter
from labels import labels
import os
import cv2
import pandas as pd


class MetricsGenerator(object):
    def __init__(self, exp_obj, eval_dataset: str, pipeline=1, depth_th=None):
        '''

        :param exp_obj:
        :param eval_dataset:
        :param depth_th: None if not considering depth,
                         [upper_th, lower_th) array of two elements if need to evaluate on the basis of depth.
        '''

        if not eval_dataset in ["gta", "cs"]:
            raise ValueError("Invalid dataset in get_gt_pred_iterator : ", eval_dataset)
        if eval_dataset == "gta" and depth_th is not None:
            raise ValueError("Depth-wise calculation with GTA dataset not supported.")

        self.exp_obj = exp_obj
        self.eval_dataset = eval_dataset

        if pipeline == 1:

            self.depth_th = depth_th

            self.labels_all = {label.name: label.id for label in labels if not label.id < 0}

            self.gt_pred_iter = self.exp_obj.get_gt_pred_iterator(eval_dataset)
            self.metric = Metric(len(self.labels_all), [l for l in self.labels_all], useUnlabeled=True)

            self.row_idx = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])
            self.conf_mat = None
            self.results_dict = None

            self.json_path = self.exp_obj.custom_eval_dir + "/resultPixelLevelSemanticLabeling_" + eval_dataset + "_custom.json"
            self.cm_path = self.exp_obj.custom_eval_dir + "/cm_" + eval_dataset + "_eval.png"

            if depth_th is not None:
                self.json_path = self.exp_obj.custom_eval_dir + "/resultPixelLevelSemanticLabeling_" + eval_dataset + "_custom_" + str(
                    depth_th[0]) + "_" + str(depth_th[1]) + ".json"
                self.cm_path = self.exp_obj.custom_eval_dir + "/cm_" + eval_dataset + "_eval_" + str(
                    depth_th[0]) + "_" + str(depth_th[1]) + ".png"

    def generate_metrics(self):
        def get_th_imgs(gt, pred, mask):
            '''

            :param gt:
            :param pred:
            :param mask:
            :return: 1D arrays where mask is True.
            '''
            mask = np.argwhere(np.equal(mask, True))
            thresholded_gt = gt[mask[:, 0], mask[:, 1]]
            thresholded_pred = pred[mask[:, 0], mask[:, 1]]

            return thresholded_gt, thresholded_pred

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

        def refine_with_depth(gt, pred, depth, upper_th, lower_th):
            mask = process_depthmap(depth, upper_th=upper_th, lower_th=lower_th)
            return get_th_imgs(gt, pred, mask)

        i = 0
        for _, gt, _, pred, _, depth, _, _ in self.gt_pred_iter:
            print("\rImages Processed: {}".format(i + 1), end=' ')
            sys.stdout.flush()

            if depth is not None and self.depth_th is not None:
                gt, pred = refine_with_depth(gt, pred, depth, self.depth_th[0], self.depth_th[1])

            self.metric.update_matrix(gt, pred)

            i += 1

            # if i>10:
            #     break
        _, _, _, _, self.conf_mat = self.metric.scores()

    def refine_cm(self):
        """removes unwanted information from confusion matrix."""
        if self.conf_mat is None:
            print("\n\t\tGenerating again")
            self.generate_metrics()

        self.metric.reset()
        # refining and getting rid of unwanted labels.
        conf_mat_summ = summaraize_cm(self.conf_mat, self.row_idx)
        ref_labels_summ = summarize_class_id_dict(self.labels_all, self.row_idx)

        self.metric = Metric(len(ref_labels_summ), [l for l in ref_labels_summ], mat=conf_mat_summ, useUnlabeled=True)

    def refine_cm_new(self):
        """removes unwanted information from confusion matrix."""
        if self.conf_mat is None:
            print("\n\t\tGenerating again")
            self.generate_metrics()

        self.metric.reset()
        # refining and getting rid of unwanted labels.
        conf_mat_summ = summaraize_cm_new(self.conf_mat, self.row_idx)
        ref_labels_summ = summarize_class_id_dict(self.labels_all, self.row_idx)

        self.metric = Metric(len(ref_labels_summ), [l for l in ref_labels_summ], mat=conf_mat_summ, useUnlabeled=True)

    def set_results_dict(self):
        ref_labels_summ = summarize_class_id_dict(self.labels_all, self.row_idx)

        accuracy, avg_accuracy, IoU, mIoU, conf_mat = self.metric.scores()

        self.results_dict = {}
        self.results_dict["averageAccuracy"] = avg_accuracy.tolist()
        self.results_dict["averageScoreClasses"] = mIoU.tolist()

        accuracy = accuracy.tolist()
        #
        classScores_summ = {}

        for label, acc in zip(ref_labels_summ, accuracy):
            classScores_summ[label] = acc

        self.results_dict["classScores"] = classScores_summ
        self.results_dict["labels"] = ref_labels_summ
        self.results_dict["confMatrix"] = conf_mat.tolist()

    def save_results(self):

        with open(self.json_path, 'w') as fp:
            json.dump(self.results_dict, fp, indent=4)

    def plot_cm(self, load_json):
        ref_labels_summ = summarize_class_id_dict(self.labels_all, self.row_idx)
        if load_json:
            with open(self.json_path) as f:
                data = json.load(f)
            confMatrix = np.asarray(data["confMatrix"])
            confMatrix = summaraize_cm(confMatrix, self.row_idx)
        else:
            confMatrix = np.asarray(self.results_dict["confMatrix"])

        Plotter.save_confusion_matrix_as_png(confMatrix, [l for l in ref_labels_summ], self.cm_path,
                                             normalize=True, cmap=plt.cm.YlGn)

    def load_all_jsons_per_exp(self, depth_dict):
        jsons = {}
        eval_dataset = "cs"
        for d in depth_dict:
            json_path = self.exp_obj.custom_eval_dir + "/resultPixelLevelSemanticLabeling_" + eval_dataset + "_custom_" + str(
                depth_dict[d][0]) + "_" + str(depth_dict[d][1]) + ".json"
            dict_key = d
            with open(json_path) as f:
                jsons[dict_key] = json.load(f)

        return jsons

    def __get_all_MIOUs(self, jsons):
        depth_MIOUs = {}
        for k in jsons:
            depth_MIOUs[k] = jsons[k]["averageScoreClasses"]

        return depth_MIOUs

    def pipeline_2(self):
        def __get_all_class_scores(jsons):
            depth_class_scores = {}
            classes = []
            for k in jsons:
                depth_class_scores[k] = jsons[k]["classScores"]

                for cl in depth_class_scores[k]:
                    if cl not in classes:
                        classes.append(cl)

            eval_dict = {}
            for k in depth_class_scores:
                eval_dict[k] = []
                for cl in classes:
                    eval_dict[k].append(depth_class_scores[k][cl])
            return eval_dict, classes

        # depth_list = [[1.0, 0.75], [0.75, 0.5], [0.5, 0.25], [0.25, 0.0]]
        depth_dict = {'very_close': [1.0, 0.75], 'close': [0.75, 0.5], 'far': [0.5, 0.25], 'very_far': [0.25, 0.0]}

        jsons = self.load_all_jsons_per_exp(depth_dict)
        # depth_MIOUs = self.__get_all_MIOUs(jsons)
        eval_dict, classes = __get_all_class_scores(jsons)

        print(eval_dict)
        Plotter.save_class_accuracy(eval_dict=eval_dict, class_names=classes, img_path="tmp.png")

    def complete_pipeline(self):
        self.generate_metrics()
        self.refine_cm()
        # self.refine_cm_new()
        self.set_results_dict()
        self.save_results()
        self.plot_cm(load_json=False)

    def load_exp_json(self):
        json_path = self.exp_obj.custom_eval_dir + "/resultPixelLevelSemanticLabeling_" + self.eval_dataset + "_custom.json"
        with open(json_path) as f:
            return json.load(f)

    def pipeline_3(self):
        json = self.load_exp_json()
        class_score_dict = json["classScores"]

        miou = json["averageScoreClasses"]
        avg_accuracy = json["averageAccuracy"]
        sorted_by_value = sorted(class_score_dict.items(), key=lambda kv: kv[1], reverse=True)
        class_score_dict = {tup[0]: tup[1] for tup in sorted_by_value}

        class_score_dict_w_color = {}
        for k in class_score_dict:
            class_score_dict_w_color[k] = {}
            class_score_dict_w_color[k]["score"] = class_score_dict[k]
            for l in labels:
                if l.name == k:
                    color = np.array(l.color, dtype=np.float64)
                    color = np.divide(color, 255)

                    class_score_dict_w_color[k]["color"] = color

        title = "Class Accuracy "
        plot_folder = self.exp_obj.custom_eval_dir
        plot_path = plot_folder + "/class_avg_" + self.eval_dataset + ".png"

        Plotter.save_class_accuracy_new(class_score_dict_w_color, title, miou, avg_accuracy, plot_path)

    def filter_by_class(self, mat, cl):
        return np.array(np.where(mat == cl, 1, 0), dtype=np.uint16)

    def calculate_true_predictions(self):

        def get_class_freq_holder():
            maxLabelID = 33
            ncols = maxLabelID + 1
            return np.zeros((1024, 2048, ncols), dtype=np.uint16)

        def get_class_frequency():
            class_frequency = get_class_freq_holder()
            elems_counter = 0
            for _, gt, _, pred, _, depth in self.gt_pred_iter:
                print("\rImages Processed: {}".format(elems_counter + 1), end=' ')
                sys.stdout.flush()

                for cl in range(34):
                    gt = gt.flatten()
                    pred = pred.flatten()

                    filter_gt_cl = self.filter_by_class(gt, cl)
                    filter_pred_cl = self.filter_by_class(pred, cl)
                    correct = np.where(filter_gt_cl == filter_pred_cl, 1, 0)

                    correct = correct.reshape(1024, 2048)

                    class_frequency[:, :, cl] = np.add(class_frequency[:, :, cl], correct)

                elems_counter += 1
                # if elems_counter > 3:
                #     break
            return class_frequency, np.array(elems_counter)

        class_frequency, elems_counter = get_class_frequency()

        data_analysis = {}

        data_analysis["class_frequency"] = class_frequency
        data_analysis["total_images"] = elems_counter

        print("\nThere are ", elems_counter)

        print("Saving")
        np.save("./data/correct_pred.npy", data_analysis)
        print("Done")

    def draw_true_predictions(self, plot_folder):
        data_analysis = np.load("./data/correct_pred.npy")[()]

        ncols = data_analysis["class_frequency"].shape[2]

        for l in labels:
            plt.clf()
            if l.ignoreInEval:
                continue
            cl = l.id
            sl = np.array(data_analysis["class_frequency"][:, :, cl], dtype=np.float64)

            total = np.array(data_analysis["total_images"], dtype=np.float64)
            class_slice = np.divide(sl, total)
            class_slice = np.divide(class_slice, ncols)

            fig = plt.figure(figsize=(20, 8))
            ax = fig.add_subplot(111)

            ax.set_title('Density for correct prediction of class : \"' + l.name)

            plt.imshow(class_slice, interpolation='nearest', origin='upper', cmap=plt.cm.Blues)
            plt.colorbar()
            plt.axis('off')
            plot_path = plot_folder + "true_" + l.name + "_density.png"
            plt.savefig(plot_path)

    def pipeline_4(self):
        # self.calculate_true_predictions()
        dir_path = os.path.dirname(os.path.realpath(__file__)) + "/data/plots/"
        self.draw_true_predictions(plot_folder=dir_path)

    def get_top2_confusing_labels_foreach(self, confMatrix):
        useful_labels = [l for l in labels if not l.ignoreInEval]

        confusion_info = {}
        for conf_id, l in enumerate(useful_labels):
            conf_args = np.argsort(confMatrix[conf_id])[::-1][:3]
            conf_args = [x for x in conf_args if x != conf_id]

            confusion_info[l.name] = {"label_info": l,
                                      "confused_by": [useful_labels[conf_args[0]], useful_labels[conf_args[1]]]}

        return confusion_info

    def pipeline_5(self):
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/plots/")
        npy_path = os.path.join(dir_path, "most_miscls_perclass_exp43.npy")

        def __find_most_misclf_in_eachClass(npy_path):
            def count_misclassification(true_class, false_class, gt, pred):
                filtered_gt_cl_1 = np.array(np.where(gt == true_class, 1, 0), dtype=np.uint16)
                filtered_pred_cl_2 = np.array(np.where(pred == false_class, 1, 0), dtype=np.uint16)

                F_cl_2 = np.where(filtered_gt_cl_1 & filtered_pred_cl_2, 1, 0)

                return np.sum(F_cl_2)

            def count_per_true_class(confusion_info_t_class, true_class, gt, pred):
                my_dict = {}
                for confused_label in confusion_info_t_class["confused_by"]:
                    false_class = confused_label.id

                    my_dict[false_class] = count_misclassification(true_class, false_class, gt, pred)
                return my_dict

            json = self.load_exp_json()
            confMatrix = json["confMatrix"]
            confusion_info = self.get_top2_confusing_labels_foreach(confMatrix)

            cols = [confusion_info[c]["label_info"].id for c in confusion_info]
            cols.append("gt_fname")
            cols.append("pred_fname")
            cols.append("rgb_fname")

            misclassification = pd.DataFrame(columns=cols)

            row_idx = 0
            for gt_file_name, gt, pred_file_name, pred, _, _, img_file_name, _ in self.gt_pred_iter:
                print("\rImages Processed: {}".format(row_idx + 1), end=' ')
                sys.stdout.flush()

                misclassification.loc[row_idx] = np.zeros(len(cols), dtype=np.uint16)

                for c_name in confusion_info:
                    true_class = confusion_info[c_name]["label_info"].id  # base class

                    misclassification.loc[row_idx][true_class] = count_per_true_class(confusion_info[c_name],
                                                                                      true_class, gt,
                                                                                      pred)
                misclassification.loc[row_idx]["gt_fname"] = gt_file_name
                misclassification.loc[row_idx]["pred_fname"] = pred_file_name
                misclassification.loc[row_idx]["rgb_fname"] = img_file_name
                row_idx += 1
                # if row_idx > 3:
                #     break

            misclassification.to_pickle(npy_path)

        def __get_paths_of_images(npy_path):

            misclassification = pd.read_pickle(npy_path)

            counter = {}
            cols = list(misclassification.columns.values)
            cols = [col for col in cols if not type(col) == str]
            # iterate through each column
            # in each cell, get

            for col_id in cols:

                # col_id = 7
                counter[col_id] = {}
                for row_idx_cell, cell_dict in enumerate(misclassification[col_id]):
                    for k in cell_dict:
                        if k not in counter[col_id]:
                            counter[col_id][k] = []
                        counter[col_id][k].append(cell_dict[k])
            best_img_dict = {}
            for true_col in counter:
                best_img_dict[true_col] = {}
                for false_col in counter[true_col]:
                    conf_args = np.argsort(counter[true_col][false_col])[::-1][:3]
                    best_img_dict[true_col][false_col] = {'rgb': [], 'pred': [], 'gt': []}

                    for arg in conf_args:
                        best_img_dict[true_col][false_col]['rgb'].append(misclassification.iloc[arg]["rgb_fname"])
                        best_img_dict[true_col][false_col]['pred'].append(misclassification.iloc[arg]["pred_fname"])
                        best_img_dict[true_col][false_col]['gt'].append(misclassification.iloc[arg]["gt_fname"])

            return best_img_dict

        def get_single_cv_image(image_file):
            """
            Returns the OpenCV image of the given filepath
            """
            image_file_path = image_file

            im = cv2.imread(image_file_path, -1)
            return im

        def create_sidebar(height, true_class_id, false_class_id):
            sidebar = np.zeros((height, 200, 3), dtype=np.uint8)

            cv2.rectangle(sidebar, (10, 50), (30, 70),
                          labels[true_class_id].color, -1)

            cv2.putText(sidebar, "True : {}".format(labels[true_class_id].name),
                        (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.rectangle(sidebar, (10, 90), (30, 110),
                          labels[false_class_id].color, -1)

            cv2.putText(sidebar, "False : {}".format(labels[false_class_id].name),
                        (40, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            return sidebar

        def create_overlay(gt_img, pred_img, rgb_img, true_class_id, false_class_id):
            filtered_gt = self.filter_by_class(gt_img, true_class_id)
            filtered_pred = self.filter_by_class(pred_img, false_class_id)

            misclassified = np.where(filtered_gt & filtered_pred, 1, 0)

            overlayed = np.zeros(shape=(misclassified.shape[0], misclassified.shape[1], 3), dtype=np.uint8)
            overlayed[:, :, :] = np.where(misclassified[:, :, np.newaxis] == True,
                                          labels[false_class_id].color,
                                          [0, 0, 0])

            output = rgb_img.copy()
            alpha = 0.4

            if labels[false_class_id].name == 'building':
                alpha = 0.7

            cv2.addWeighted(overlayed, alpha, rgb_img, 1,
                            0, output)

            sidebar = create_sidebar(output.shape[0], true_class_id, false_class_id)
            return np.hstack((output, sidebar))

        def paint_and_save_false_classes(img_paths, true_class_id, false_class_id, img_dir):
            rgb_img_paths = img_paths[true_class_id][false_class_id]['rgb']
            pred_img_paths = img_paths[true_class_id][false_class_id]['pred']
            gt_img_paths = img_paths[true_class_id][false_class_id]['gt']

            for id, rgb_path in enumerate(rgb_img_paths):
                pred_path = pred_img_paths[id]
                gt_path = gt_img_paths[id]

                rgb_img = get_single_cv_image(rgb_path)
                gt_img = get_single_cv_image(gt_path)
                pred_img = get_single_cv_image(pred_path)

                painted = create_overlay(gt_img, pred_img, rgb_img, true_class_id, false_class_id)
                folder_path = os.path.join(img_dir, labels[true_class_id].name + "/")
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_path = os.path.join(folder_path,
                                         labels[true_class_id].name + "_as_" + labels[false_class_id].name + "_" + str(
                                             id) + ".png")

                cv2.imwrite(file_path, painted)

        # __find_most_misclf_in_eachClass(npy_path)  # in order to save name of images which are most mis-classified
        img_paths = __get_paths_of_images(npy_path)

        painted_imags_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/painted/")

        for true_class_id in img_paths:
            for false_class_id in img_paths[true_class_id]:
                paint_and_save_false_classes(img_paths, true_class_id, false_class_id, painted_imags_path)
        # exit()


def get_file_names(dir_name):
    only_files = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
    only_files.sort()
    return only_files


def summaraize_cm(cm, indices_to_consider):
    return cm[indices_to_consider[:, None], indices_to_consider]


def summaraize_cm_new(cm, indices_to_consider):
    col_idx = np.arange(cm.shape[1])
    indices_to_not_consider = np.array([c for c in col_idx if c not in indices_to_consider])

    mat_consider = cm[indices_to_consider[:, None], indices_to_consider]

    false_others = np.sum(cm[indices_to_consider[:, None], indices_to_not_consider], axis=1)
    false_others = false_others.reshape(false_others.shape[0], 1)

    true_others = np.sum(cm[indices_to_not_consider[:, None], indices_to_not_consider])
    false_positives = cm[indices_to_not_consider[:, None], indices_to_consider]
    false_positives = np.sum(false_positives, axis=0)
    final_mat = np.hstack((mat_consider, false_others))
    others = np.hstack((false_positives, true_others))
    final_mat = np.vstack((final_mat, others))

    return final_mat


def summarize_class_id_dict(class_id_dict: dict, indices_to_consider):
    ref_labels_summ = {k: v for (k, v) in class_id_dict.items() if class_id_dict[k] in indices_to_consider}
    # ref_labels_summ['others'] = 34
    return ref_labels_summ


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

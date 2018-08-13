import os
from JSONResults.results import *
from Plotter import Plotter
from os import listdir
from os.path import isfile, join
from Iterator.GT_Pred_Iterator import GT_Pred_Iterator
from Evaluation.labels import *
from MetricsGeneratorPkg.Metric import Metric
from MetricsGeneratorPkg.MetricsGenerator import MetricsGenerator


class Experiment(object):

    def __init__(self, exp_path, special=None):

        self.exp_path = exp_path
        self.special = special

        self.train_dir = None
        self.eval_cityscapes_dir = None
        self.eval_gta_dir = None
        self.vis_cityscapes_dir = None
        self.vis_gta_dir = None

        self.custom_eval_dir = None
        self.initialize_dir_vars()

        self.__create_dir(self.custom_eval_dir)

        self.custom_cityscapes_eval_json_results = None
        self.custom_gta_eval_json_results = None

    def save_all_results_pngs(self):
        Plotter.save_confusion_matrix_as_png(self.custom_cityscapes_eval_json_results.confusion_matrix,
                                             self.custom_cityscapes_eval_json_results.labels,
                                             self.custom_eval_dir + '/cm_cs_eval.png',
                                             normalize=True, title="Evaluated on Cityscapes")
        Plotter.save_confusion_matrix_as_png(self.custom_gta_eval_json_results.confusion_matrix,
                                             self.custom_gta_eval_json_results.labels,
                                             self.custom_eval_dir + '/cm_gta_eval.png',
                                             normalize=True, title="Evaluated on GTA")

    def load_custom_eval(self):
        print("Need to change JSONResults in load_custom_eval()")
        exit()
        if self.__file_exist(self.custom_eval_dir + "/resultPixelLevelSemanticLabeling_cityscapes_custom.json"):
            self.custom_cityscapes_eval_json_results = JSONResults(
                results_loc=self.custom_eval_dir + "/resultPixelLevelSemanticLabeling_cityscapes_custom.json")
        else:
            print("\n\t File do not exist: ",
                  self.custom_eval_dir + "/resultPixelLevelSemanticLabeling_cityscapes_custom.json")

        if self.__file_exist(self.custom_eval_dir + "/resultPixelLevelSemanticLabeling_gta_custom.json"):
            self.custom_gta_eval_json_results = JSONResults(
                results_loc=self.custom_eval_dir + "/resultPixelLevelSemanticLabeling_gta_custom.json")
        else:
            print("\n\t File do not exist: ",
                  self.custom_eval_dir + "/resultPixelLevelSemanticLabeling_gta_custom.json")

    def get_gt_pred_iterator(self, eval_dataset="cs"):
        print("Need to handle")
        exit()
        if not eval_dataset in ["gta", "cs"]:
            raise ValueError("Invalid dataset in get_gt_pred_iterator : ", eval_dataset)
        gt_loc = ""
        pred_loc = self.vis_cityscapes_dir

        if eval_dataset == "gta":
            gt_loc = ""
            pred_loc = self.vis_cityscapes_dir
        gt_paths = self.get_file_names(gt_loc)
        pred_paths = self.get_file_names(pred_loc)

        gt_pred_iter = GT_Pred_Iterator(ground_truth_filenames=gt_paths, prediction_filenames=pred_paths)

        return gt_pred_iter

    def evaluate_cs(self):
        gt_pred_iter = self.get_gt_pred_iterator(eval_dataset="cs")
        labels_all = {label.name: label.id for label in labels if not label.id < 0}
        metric = Metric(len(labels_all), [l for l in labels_all], useUnlabeled=True)
        row_idx = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])

        MetricsGenerator.saveJSON(iterator=gt_pred_iter, metric=metric,
                                  json_path=self.custom_eval_dir + "/resultPixelLevelSemanticLabeling_cityscapes_custom.json",
                                  ref_labels=labels_all, row_idx=row_idx)

    def evaluate_gta(self):
        gt_pred_iter = self.get_gt_pred_iterator(eval_dataset="gta")
        labels_all = {label.name: label.id for label in labels if not label.id < 0}
        metric = Metric(len(labels_all), [l for l in labels_all], useUnlabeled=True)
        row_idx = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])

        MetricsGenerator.saveJSON(iterator=gt_pred_iter, metric=metric,
                                  json_path=self.custom_eval_dir + "/resultPixelLevelSemanticLabeling_gta_custom.json",
                                  ref_labels=labels_all, row_idx=row_idx)

    def initialize_dir_vars(self):
        self.train_dir = self.exp_path + '/train'

        self.eval_cityscapes_dir = self.exp_path + '/eval_cityscapes'
        self.eval_gta_dir = self.exp_path + '/eval_gta'
        self.vis_cityscapes_dir = self.exp_path + '/vis_cityscapes/raw_segmentation_results'
        self.vis_gta_dir = self.exp_path + '/vis_gta/raw_segmentation_results'

        self.custom_eval_dir = self.exp_path + '/custom_eval'

        self.check_all_dirs_exist()  # raises Value Error if all but custom_eval dir do not exist in an experiment.

    def __if_dir_exist(self, my_dir):
        return os.path.isdir(my_dir)

    def __file_exist(self, file_path):
        return os.path.isfile(file_path)

    def __create_dir(self, my_dir):
        if not os.path.exists(my_dir):
            os.makedirs(my_dir)

    def check_all_dirs_exist(self):
        if not self.train_exist() or not self.eval_cityscapes_exist() or not self.eval_gta_exist() \
                or not self.vis_cityscapes_exist() or not self.vis_gta_exist():
            raise ValueError("Some of the directories not exist.")

    def train_exist(self):
        return self.__if_dir_exist(self.train_dir)

    def eval_cityscapes_exist(self):
        return self.__if_dir_exist(self.eval_cityscapes_dir)

    def eval_gta_exist(self):
        return self.__if_dir_exist(self.eval_gta_dir)

    def vis_cityscapes_exist(self):
        return self.__if_dir_exist(self.vis_cityscapes_dir)

    def vis_gta_exist(self):
        return self.__if_dir_exist(self.vis_gta_dir)

    @staticmethod
    def get_file_names(dir_name):
        only_files = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
        only_files.sort()
        return only_files

import os
from JSONResults.results import *
from Plotter import Plotter
from os import listdir
from os.path import isfile, join
from Iterator.GT_Pred_Iterator import GT_Pred_Iterator
import glob


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

    @staticmethod
    def get_cs_gt_filepaths():
        server = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
        cs_root_dir = server + '/home/mumu01/models/deeplab/datasets/cityscapes'
        groundTruthSearch = os.path.join(cs_root_dir, "gtFine", "val", "*", "*_gtFine_labelIds.png")
        groundTruthImgList = glob.glob(groundTruthSearch)
        groundTruthImgList.sort()
        return groundTruthImgList

    @staticmethod
    def get_cs_img_filepaths():
        server = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
        cs_root_dir = server + '/home/mumu01/models/deeplab/datasets/cityscapes'
        groundTruthSearch = os.path.join(cs_root_dir, "leftImg8bit", "val", "*", "*_leftImg8bit.png")
        groundTruthImgList = glob.glob(groundTruthSearch)
        groundTruthImgList.sort()
        return groundTruthImgList

    @staticmethod
    def get_cs_disparity_filepaths():
        server = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
        cs_root_dir = server + '/home/mumu01/models/deeplab/datasets/cityscapes'
        disparitySearch = os.path.join(cs_root_dir, "disparity", "val", "*", "*_disparity.png")
        disparityImgList = glob.glob(disparitySearch)
        disparityImgList.sort()
        return disparityImgList

    @staticmethod
    def get_gta_gt_filepaths(self):
        server = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
        gta_root_dir = server + '/home/mumu01/models/deeplab/datasets/gta/Ids'
        pred_gta = self.vis_gta_dir
        pred_filenames = self.get_file_names(pred_gta)

        groundTruthImgList = []

        for filePath in pred_filenames:
            f = os.path.basename(filePath)
            groundTruthImgList.append(gta_root_dir + '/' + f)
        groundTruthImgList.sort()
        return groundTruthImgList

    def get_gt_pred_iterator(self, eval_dataset="cs"):

        if not eval_dataset in ["gta", "cs"]:
            raise ValueError("Invalid dataset in get_gt_pred_iterator : ", eval_dataset)

        gt_paths = self.get_cs_gt_filepaths()
        disparity_paths = self.get_cs_disparity_filepaths()
        pred_loc = self.vis_cityscapes_dir
        img_paths = self.get_cs_img_filepaths()

        if eval_dataset == "gta":
            gt_paths = self.get_gta_gt_filepaths(self)
            pred_loc = self.vis_gta_dir
            disparity_paths = None

        pred_paths = self.get_file_names(pred_loc)

        gt_pred_iter = GT_Pred_Iterator(ground_truth_filenames=gt_paths, prediction_filenames=pred_paths,
                                        disparity_filenames=disparity_paths, image_paths=img_paths,
                                        dataset=eval_dataset)

        return gt_pred_iter

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


if __name__ == '__main__':
    server = "/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01"
    exps_root_dir = server + "/home/mumu01/exps"
    exps = ["55"]

    for exp_num in exps:
        re_eval = False
        exp_dir = exps_root_dir + '/exp' + exp_num
        exp_obj = Experiment(exp_dir)
        exp_obj.load_custom_eval()
        exp_obj.save_all_results_pngs()

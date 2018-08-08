import os


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

    def initialize_dir_vars(self):
        self.train_dir = self.exp_path + '/train'

        if self.special is None:
            self.eval_cityscapes_dir = self.exp_path + '/eval_cityscapes'
            self.eval_gta_dir = self.exp_path + '/eval_gta'
            self.vis_cityscapes_dir = self.exp_path + '/vis_cityscapes/raw_segmentation_results'
            self.vis_gta_dir = self.exp_path + '/vis_gta/raw_segmentation_results'

            self.custom_eval_dir = self.exp_path + '/custom_eval'
        else:
            raise ValueError('Need to handle special cases.')

    def __if_dir_exist(self, my_dir):
        return os.path.isdir(my_dir)

    def __create_dir(self, my_dir):
        if not os.path.exists(my_dir):
            os.makedirs(my_dir)

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
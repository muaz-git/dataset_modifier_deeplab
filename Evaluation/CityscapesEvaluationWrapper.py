import os
from Experiment import Experiment


class CityscapesEvaluationWrapper(object):

    def __init__(self, exp_obj: Experiment, eval_dataset: str):
        self.script_path = '/home/mumu01/models/deeplab/datasets/cityscapes/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py'
        self.exp_obj = exp_obj

        self.eval_dataset = eval_dataset

        if not (eval_dataset in ['gta', 'cityscapes']):
            raise ValueError(eval_dataset + ' does not exist.')

    def evaluate_using_CS_script(self):

        if not self.__is_script_valid():
            raise ValueError(self.script_path + ' is not a valid script.')

        export_file = self.exp_obj.custom_eval_dir + '/resultPixelLevelSemanticLabeling_' + self.eval_dataset + '.json'
        ID_dir = self.exp_obj.vis_cityscapes_dir
        log_file = self.exp_obj.custom_eval_dir + '/wrapper_' + self.eval_dataset + '_logs.txt'

        if self.eval_dataset == 'gta':
            ID_dir = self.exp_obj.vis_gta_dir + ' -g'

        complete_script = 'nohup ' + self.script_path + ' -e ' + export_file + ' -t ' + ID_dir + ' > ' + log_file + ' &'

        os.system(complete_script)

    def __file_exist(self, file_path):
        return os.path.isfile(file_path)

    def __is_script_valid(self):
        return self.__file_exist(self.script_path)


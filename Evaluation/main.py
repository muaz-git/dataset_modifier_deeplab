import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Experiment import Experiment
from CityscapesEvaluationWrapper import CityscapesEvaluationWrapper
import json
import os
import numpy as np
from Plotter import Plotter
from JSONResults.results import *
from Evaluation.labels import *
from MetricsGeneratorPkg.Metric import Metric
from MetricsGeneratorPkg.MetricsGenerator import MetricsGenerator

exps_root_dir = "/home/mumu01/exps"
exps = ["53", "54"]

# for exp_num in exps:
# # exp_num = "53"
#
#     re_eval = False
#     exp_dir = exps_root_dir + '/exp' + exp_num
#
#     exp_obj = Experiment(exp_dir)
#     exp_obj.load_custom_eval()
#     exp_obj.save_all_results_pngs()

# evaluator = CityscapesEvaluationWrapper(exp_obj, "cityscapes")
# if re_eval:

# print("Evaluating for cityscapes")
# evaluator.evaluate_using_CS_script()

# evaluator = CityscapesEvaluationWrapper(exp_obj, "gta")
# if re_eval:
#     print("Evaluating for GTA")
#     evaluator.evaluate_using_CS_script(is_gta=True)

exp_num = "53"
exp_dir = exps_root_dir + '/exp' + exp_num
exp_obj = Experiment(exp_dir)
gt_pred_iter = exp_obj.get_gt_pred_iterator()
labels_all = {label.name: label.id for label in labels if not label.id < 0}
metric = Metric(len(labels_all), [l for l in labels_all], useUnlabeled=True)

row_idx = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])

MetricsGenerator.saveJSON(iterator=gt_pred_iter, metric=metric, json_path="results_54_cs.json",
                          ref_labels=labels_all, row_idx=row_idx)

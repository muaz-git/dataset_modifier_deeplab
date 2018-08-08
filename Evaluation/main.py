from Experiment import Experiment
from CityscapesEvaluationWrapper import CityscapesEvaluationWrapper

exps_root_dir = "/home/mumu01/exps"

exp_num = "53"


exp_dir = exps_root_dir + '/exp' + exp_num

exp_obj = Experiment(exp_dir)

evaluator = CityscapesEvaluationWrapper(exp_obj, "cityscapes")
print("Evaluating for cityscapes")
evaluator.evaluate_using_CS_script()
evaluator = CityscapesEvaluationWrapper(exp_obj, "gta")
print("Evaluating for GTA")
evaluator.evaluate_using_CS_script()
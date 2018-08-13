import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from os import listdir
from os.path import isfile, join
import os
import numpy as np
from sklearn.metrics import confusion_matrix

from collections import namedtuple

Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).

    'trainId',  # An integer ID that overwrites the ID above, when creating ground truth
    # images for training.
    # For training, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    # Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label('static', 4, 255, 'void', 0, False, True, (20, 20, 20)),  # in gta dataset it has a different color
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'ground', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'ground', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'ground', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'ground', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]


class Evaluator(object):
    def __init__(self):
        pass


class CityscapesEvaluator(Evaluator):
    def __init__(self, **kwargs):
        Evaluator.__init__(self)
        self.ground_truth_filenames = kwargs["ground_truth_filenames"]
        self.prediction_filenames = kwargs["prediction_filenames"]

        if not len(self.ground_truth_filenames) == len(self.prediction_filenames):
            raise ValueError('Length of files is not equal.')

        self.iter_index = 0

    @staticmethod
    def get_single_cv_image(image_file):
        """
        Returns the OpenCV image of the given filepath
        """
        image_file_path = image_file

        im = cv2.imread(image_file_path, 0)
        return im

    def __iter__(self):
        return self

    def __next__(self):

        if self.iter_index < len(self.ground_truth_filenames):
        # if self.iter_index < 5:
            gt_file_name = self.ground_truth_filenames[self.iter_index]
            pred_file_name = self.prediction_filenames[self.iter_index]

            gt_image = self.get_single_cv_image(gt_file_name)
            pred__image = self.get_single_cv_image(pred_file_name)
            self.iter_index += 1
            return gt_file_name, gt_image, pred_file_name, pred__image
        else:
            raise StopIteration()

class ConfusionMatrix:
    def __init__(self, nclasses, classes, useUnlabeled=False):
        self.mat = np.zeros((nclasses, nclasses), dtype=np.float)
        self.valids = np.zeros((nclasses), dtype=np.float)
        self.IoU = np.zeros((nclasses), dtype=np.float)
        self.mIoU = 0

        self.nclasses = nclasses
        self.classes = classes
        self.list_classes = list(range(nclasses))
        self.useUnlabeled = useUnlabeled
        self.matStartIdx = 1 if not self.useUnlabeled else 0

    def update_matrix(self, target, prediction):
        if not (isinstance(prediction, np.ndarray)) or not (isinstance(target, np.ndarray)):
            print("Expecting ndarray")
        elif len(target.shape) == 3:  # batched spatial target
            if len(prediction.shape) == 4:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 3:
                temp_prediction = prediction.flatten()
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target.flatten()
        elif len(target.shape) == 2:  # spatial target
            if len(prediction.shape) == 3:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 2:
                temp_prediction = prediction.flatten()
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target.flatten()
        elif len(target.shape) == 1:
            if len(prediction.shape) == 2:  # prediction is 1 hot encoded
                temp_prediction = np.argmax(prediction, axis=1).flatten()
            elif len(prediction.shape) == 1:
                temp_prediction = prediction
            else:
                print("Make sure prediction and target dimension is correct")

            temp_target = target
        else:
            print("Data with this dimension cannot be handled")

        self.mat += confusion_matrix(temp_target, temp_prediction, labels=self.list_classes)

    def scores(self):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        total = 0  # Total true positives
        N = 0  # Total samples
        for i in range(self.matStartIdx, self.nclasses):
            N += sum(self.mat[:, i])
            tp = self.mat[i][i]
            fp = sum(self.mat[self.matStartIdx:, i]) - tp
            fn = sum(self.mat[i, self.matStartIdx:]) - tp

            if (tp + fp) == 0:
                self.valids[i] = 0
            else:
                self.valids[i] = tp / (tp + fp)

            if (tp + fp + fn) == 0:
                self.IoU[i] = 0
            else:
                self.IoU[i] = tp / (tp + fp + fn)

            total += tp

        self.mIoU = sum(self.IoU[self.matStartIdx:]) / (self.nclasses - self.matStartIdx)
        self.accuracy = total / (sum(sum(self.mat[self.matStartIdx:, self.matStartIdx:])))

        return self.valids, self.accuracy, self.IoU, self.mIoU, self.mat

    def plot_confusion_matrix(self, filename):
        # Plot generated confusion matrix
        print(filename)

    def reset(self):
        self.mat = np.zeros((self.nclasses, self.nclasses), dtype=float)
        self.valids = np.zeros((self.nclasses), dtype=float)
        self.IoU = np.zeros((self.nclasses), dtype=float)
        self.mIoU = 0

def get_file_names(dir_name):
    only_files = [join(dir_name, f) for f in listdir(dir_name) if isfile(join(dir_name, f))]
    only_files.sort()
    return only_files


if __name__ == '__main__':
    from Plotter import Plotter
    from JSONResults.results import *
    base_loc = '../datasets'

    dataset_loc = base_loc + '/cityscapes'

    gt_loc = dataset_loc + '/groundtruth'
    pred_loc = dataset_loc + '/predictions'

    gt_paths = get_file_names(gt_loc)
    pred_paths = get_file_names(pred_loc)

    cEvaluator = CityscapesEvaluator(ground_truth_filenames=gt_paths, prediction_filenames=pred_paths)
    # gt_file_name, gt_image, pred_file_name, pred__image = cEvaluator.__next__()
    i = 1
    # for gt, _, pred, _ in cEvaluator:
    #     print(i)
    #     if not os.path.basename(gt) == os.path.basename(pred):
    #         raise ValueError("mismatch")
    #
    #     i+=1
    # print(np.shape(pred__image))

    # a label and all meta information
    # print(len(labels))
    # exit()
    considered_labels = [label for label in labels if not (label.ignoreInEval)]
    class_names_list = [label.name for label in labels]

    metrics = ConfusionMatrix(len(class_names_list), class_names_list,useUnlabeled=True)
    import sys
    i = 0
    # for _, gt, _, pred in cEvaluator:
    #     print("\rImages Processed: {}".format(i + 1), end=' ')
    #     sys.stdout.flush()
    #     i+=1
    #     metrics.update_matrix(gt, pred)
    #
    # accuracy, avg_accuracy, IoU, mIoU, conf_mat = metrics.scores()

    json_file = "../../resultPixelLevelSemanticLabeling_cityscapes.json"

    json_results = JSONResults(results_loc=json_file)
    print(json_results.labels)
    # Plotter.save_confusion_matrix_as_png(conf_mat, json_results.labels, './testing_other_cityscapes.png',
    #                                      normalize=True, cmap=plt.cm.YlGn)
    # print("accuracy : ", (accuracy))
    # print("avg_accuracy : ", (avg_accuracy))
    # print("IoU : ", (IoU))
    # print("mIoU : ", (mIoU))
    # print("conf_mat : ", np.shape(conf_mat))
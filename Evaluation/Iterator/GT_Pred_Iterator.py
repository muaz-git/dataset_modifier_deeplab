import cv2
import numpy as np


class GT_Pred_Iterator(object):
    def __init__(self, **kwargs):
        self.ground_truth_filenames = kwargs["ground_truth_filenames"]
        self.prediction_filenames = kwargs["prediction_filenames"]
        self.disparity_filenames = kwargs["disparity_filenames"]
        self.image_filenames = kwargs["image_paths"]
        self.dataset = kwargs["dataset"]

        if not self.dataset in ["gta", "cs"]:
            raise ValueError('Dataset does not exist.')

        if not len(self.ground_truth_filenames) == len(self.prediction_filenames) or not len(
                self.prediction_filenames) == len(self.image_filenames):
            print(len(self.ground_truth_filenames))
            print(len(self.prediction_filenames))
            print(len(self.image_filenames))
            raise ValueError('Length of files is not equal.')

        self.iter_index = 0

    @staticmethod
    def get_single_cv_image(image_file):
        """
        Returns the OpenCV image of the given filepath
        """
        image_file_path = image_file

        im = cv2.imread(image_file_path, -1)
        return im

    def __iter__(self):
        return self

    def __next__(self):

        if self.iter_index < len(self.ground_truth_filenames):
            # if self.iter_index < 5:
            gt_file_name = self.ground_truth_filenames[self.iter_index]
            pred_file_name = self.prediction_filenames[self.iter_index]
            img_file_name = self.image_filenames[self.iter_index]

            disparity_file_name = None
            disparity_image = None

            if self.disparity_filenames is not None:
                disparity_file_name = self.prediction_filenames[self.iter_index]
                disparity_image = self.get_single_cv_image(disparity_file_name)

            gt_image = self.get_single_cv_image(gt_file_name)
            pred_image = self.get_single_cv_image(pred_file_name)
            rgb_image = self.get_single_cv_image(img_file_name)

            self.iter_index += 1

            return gt_file_name, gt_image, pred_file_name, pred_image, disparity_file_name, disparity_image, img_file_name, rgb_image

        else:
            raise StopIteration()

import cv2


class GT_Pred_Iterator(object):
    def __init__(self, **kwargs):
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

import json
import operator
import numpy as np


class Results(object):
    def __init__(self):
        pass


class JSONResults(Results):
    def __init__(self, **kwargs):
        Results.__init__(self)
        self.json_loc = kwargs["results_loc"]
        self.json_data = self.__load_json()

        self.classes_ID_dict = dict(self.get_class_ID_dict())
        self.confusion_matrix = np.asarray(self.json_data["confMatrix"])
        self.average_score_classes = self.json_data["averageScoreClasses"]

        # indices to consider
        row_idx = np.array([7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33])

        self.confusion_matrix= self.summaraize_cm(self.confusion_matrix, row_idx)

        # self.classes_ID_dict = self.summarize_class_id_dict(self.classes_ID_dict, row_idx)
        # self.classes_ID_dict = {k:v for k,v in self.classes_ID_dict.items()}
        self.labels = self.get_labels(self.classes_ID_dict)

    def __load_json(self):
        with open(self.json_loc) as f:
            json_data = json.load(f)
        return json_data

    def get_class_ID_dict(self):
        return dict(self.json_data["labels"])

    def summaraize_cm(self, cm, indices_to_consider):
        return cm[indices_to_consider[:, None], indices_to_consider]

    def summarize_class_id_dict(self, class_id_dict: dict, indices_to_consider):
        return {k: v for (k, v) in class_id_dict.items() if class_id_dict[k] in indices_to_consider}

    def get_labels(self, class_id_dict: dict):
        return [label for (label, id) in class_id_dict.items()]

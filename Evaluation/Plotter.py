import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import numpy as np

class Plotter(object):
    def __init__(self):
        pass

    @staticmethod
    def save_confusion_matrix_as_png(confusion_matrix, classes, dest_loc, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
                This function prints and plots the confusion matrix.
                Normalization can be applied by setting `normalize=True`.
                """
        plt.clf()
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure(figsize=(20, 14))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(dest_loc)

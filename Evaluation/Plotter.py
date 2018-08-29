import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import numpy as np


class Plotter(object):
    def __init__(self):
        pass

    @staticmethod
    def save_confusion_matrix_as_png(confusion_matrix, classes, dest_loc, normalize=False, title='Confusion matrix',
                                     cmap=plt.cm.Blues):
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

    @staticmethod
    def draw_and_save(eval_dict: dict, x_axis_names, img_path, fig_size, label_dict, angle=0, method=1):
        plt.clf()
        w = 0.2

        x = np.asarray([n for n in range(len(x_axis_names))], dtype=np.float64)

        plt.figure(figsize=fig_size)
        ax = plt.subplot(111)

        if method == 2:
            for ix, e, col in (zip(range(len(eval_dict)), eval_dict, ['r', 'g', 'b', 'y'])):
                t = ix * w
                ax.bar(x + t - w / 4, eval_dict[e], width=w, color=col, align='center', label=e)
        else:

            ax.bar(x - w / 2, eval_dict["CS"], width=w, color='r', align='center', label='CS')
            ax.bar(x + w / 2, eval_dict["GTA"], width=w, color='g', align='center', label='GTA')

        ax.legend(loc='best')

        ax.set_xticks(x)
        ax.set_xticklabels(x_axis_names, rotation=angle)
        if not method == 2:
            ax.set_ylim([0, 1])

        ax.set_xlabel(label_dict["x"])
        ax.set_ylabel(label_dict["y"])

        ax.set_title(label_dict["title"])
        plt.savefig(img_path)

    @staticmethod
    def save_miou(eval_dict: dict, exp_names, img_path):
        fig_size = (20, 14)
        x_label = "Experiments"
        y_label = "MIOU"
        label_dict = {"x": x_label, "y": y_label, "title": "Evaluation score"}

        Plotter.draw_and_save(eval_dict, exp_names, img_path, fig_size, label_dict)

    @staticmethod
    def save_class_accuracy(eval_dict: dict, class_names, img_path):
        fig_size = (20, 10)
        x_label = "Classes"
        y_label = "Accuracy"
        label_dict = {"x": x_label, "y": y_label, "title": "Evaluation score"}

        Plotter.draw_and_save(eval_dict, class_names, img_path, fig_size, label_dict, angle=30, method=2)

    @staticmethod
    def save_class_accuracy_new(class_dict, title, miou, avg_accuracy, img_path):
        def autolabel(rects, xtick_labels):
            """
            Attach a text label above each bar displaying its height
            """
            for rect, l in zip(rects, xtick_labels):
                height = rect.get_height()

                ax.text(rect.get_x() + rect.get_width() / 2., height + 0.01,
                        '%s' % l,
                        ha='center', va='bottom', rotation="vertical")

        class_scores = [class_dict[k]["score"] for k in class_dict]
        class_colors = [class_dict[k]["color"] for k in class_dict]
        class_names = [k for k in class_dict]

        plt.clf()
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)

        ax.set_title(title)
        N = len(class_scores)
        margin = 0.25
        width = 3

        ind = np.arange(N)

        ind = [idx * margin + (0.5 * width + idx * width) for idx in ind]

        # handling average accuracy.
        # ind.append((N-1) * 1.5 * margin + (0.5 * width + (N-1) * width))
        # class_scores.append(avg_accuracy)
        # class_names.append("Avg. Accuracy")
        # class_colors.append([0.85, 0.5, 0.5])
        ax.axhline(y=avg_accuracy, color=[0.65, 0.5, 0.5], linestyle='dashdot', label="Avg. Class Accuracy")

        # handling MIOU.
        ind.append((N) * 1.5 * margin + (0.5 * width + (N) * width))
        class_scores.append(miou)
        class_names.append("MIOU")
        class_colors.append([0.5, 0.5, 0.5])

        ax.bar(ind, class_scores, width, color=class_colors)
        # autolabel(rects1, class_names)

        # ax.set_ylim([0, 0.5])
        ax.legend(loc='best')
        ax.set_xticks(ind)
        ax.set_xticklabels(class_names, rotation=45)

        ax.set_xlabel("Classes")
        ax.set_ylabel("Accuracy")

        # plot_folder = "./data/plots/class_average/"
        # plot_path = plot_folder + split_name + "_class_avg.png"
        # if use_eval:
        #     plot_path = plot_folder + split_name + "_class_avg_filtered.png"
        plt.savefig(img_path)


if __name__ == '__main__':
    # exp_names = ["exp53", "exp54"]
    # eval = {"CS": [0.548759341, 0.413388133], "GTA": [0.249188319, 0.359475583]}
    # Plotter.save_miou(eval_dict=eval, exp_names=exp_names, img_path="tmp.png")

    total_classes = 19
    gta = np.random.rand(total_classes)
    gta_1 = np.random.rand(total_classes)
    cs = np.random.rand(total_classes)
    cs_1 = np.random.rand(total_classes)
    eval = {"CS": np.asarray(cs), "GTA": np.asarray(gta), "GTA_1": np.asarray(gta_1), "CS_1": np.asarray(cs_1)}

    class_names = ["class_" + str(c + 1) for c in range(total_classes)]
    Plotter.save_class_accuracy(eval_dict=eval, class_names=class_names, img_path="tmp.png")

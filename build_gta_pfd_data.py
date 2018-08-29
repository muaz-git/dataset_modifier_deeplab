# val : 2500 10% of total
# train : 14875 60% of total
# test : 7625 30%
# total : 25000
import tensorflow as tf
import re
from os import listdir
from os.path import isfile, join
import math
import build_data
import os
import numpy as np
import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('img_dir_name',
                           './gta/images/',
                           'GTA label files\'s root.')

tf.app.flags.DEFINE_string('trainIDs_dir_name',
                           './gta/trainIDs/',
                           'path where encoded label files are saved.')

tf.app.flags.DEFINE_string(
    'output_dir',
    './gta/tfrecord/',
    'Path to save converted SSTable of TensorFlow examples.')

gta_pfd_root = ''
output_dir = ''
_NUM_SHARDS = 10


def get_file_names(dir_name):
    only_files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    only_files.sort()
    return np.asarray(only_files)


def split_train_val_sets():
    np.random.seed(0)  # setting seed to 0 for regeneration

    val_percent = 0.1
    img_dir_name = FLAGS.img_dir_name
    labels_dir_name = FLAGS.trainIDs_dir_name

    image_files = get_file_names(img_dir_name)
    label_files = get_file_names(labels_dir_name)

    # fixing number of validation images.
    number_of_val_images = 2500  # int(val_percent * len(image_files))

    indices = np.random.permutation(len(image_files))

    valIdx, trainIdx = indices[:number_of_val_images], indices[number_of_val_images:]

    image_files_train, image_files_val = image_files[trainIdx], image_files[valIdx]
    label_files_train, label_files_val = label_files[trainIdx], label_files[valIdx]

    my_dict = {}
    my_dict['train'] = (image_files_train.tolist(), label_files_train.tolist())
    my_dict['val'] = (image_files_val.tolist(), label_files_val.tolist())

    return my_dict


def _convert_dataset(dataset_split, dataset):
    """Converts the specified dataset split to TFRecord format.


    Raises:
      RuntimeError: If loaded image and label have different shape, or if the
        image file with specified postfix could not be found.
    """

    img_dir_name = FLAGS.img_dir_name
    labels_dir_name = FLAGS.trainIDs_dir_name
    output_dir = FLAGS.output_dir

    image_files = dataset[0]
    label_files = dataset[1]

    if not (len(image_files) == len(label_files)):
        raise RuntimeError('Length mismatch image and label.')

    num_images = len(image_files)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('png', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        shard_filename = '%s-%05d-of-%05d.tfrecord' % (
            dataset_split, shard_id, _NUM_SHARDS)
        output_filename = os.path.join(output_dir, shard_filename)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                # Read the image.
                image_data = tf.gfile.FastGFile(img_dir_name + image_files[i], 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_data = tf.gfile.FastGFile(labels_dir_name + label_files[i], 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    print("Shape mismatched between image and label. height. Ignoring.")
                    continue
                    raise RuntimeError('Shape mismatched between image and label. height : ', height, ' seg_height: ',
                                       seg_height, ' width: ', width, ' seg_width: ', seg_width, ' \nlabel_files[i]: ',
                                       label_files[i], ' image_files[i]: ', image_files[i])
                # Convert to tf example.

                if not (image_files[i] == label_files[i]):
                    raise RuntimeError(
                        'image file name : ' + image_files[i] + ' is not equal to label file name : ' + label_files[i])
                filename = os.path.basename(image_files[i])

                example = build_data.image_seg_to_tfexample(
                    image_data, filename, height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
    # Only support converting 'train' and 'val' sets for now.
    #
    my_dict = split_train_val_sets()

    for dataset_split in ['train', 'val']:
        print("converting : " + dataset_split + " set.")
        _convert_dataset(dataset_split, my_dict[dataset_split])


if __name__ == '__main__':
    tf.app.run()

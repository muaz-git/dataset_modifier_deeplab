import tensorflow as tf
import re
from os import listdir
from os.path import isfile, join
import math
import build_data
import os

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
    return only_files


def _convert_dataset():
    """Converts the specified dataset split to TFRecord format.


    Raises:
      RuntimeError: If loaded image and label have different shape, or if the
        image file with specified postfix could not be found.
    """
    dataset_split = 'train'
    img_dir_name = FLAGS.img_dir_name
    labels_dir_name = FLAGS.trainIDs_dir_name
    output_dir = FLAGS.output_dir

    image_files = get_file_names(img_dir_name)
    label_files = get_file_names(labels_dir_name)

    if not(len(image_files) == len(label_files)):
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
                image_data = tf.gfile.FastGFile(img_dir_name+image_files[i], 'rb').read()
                height, width = image_reader.read_image_dims(image_data)
                # Read the semantic segmentation annotation.
                seg_data = tf.gfile.FastGFile(labels_dir_name+label_files[i], 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
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
    # for dataset_split in ['train', 'val']:
    _convert_dataset()


if __name__ == '__main__':
    tf.app.run()

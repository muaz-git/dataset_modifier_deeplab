import numpy as np
import cv2
from scipy.io import loadmat
from os import listdir
from os.path import isfile, join
import os
from collections import namedtuple
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('gta_labels',
                           './gta/labels/',
                           'GTA label files\'s root.')

tf.app.flags.DEFINE_string(
    'trainID_dir',
    './gta/trainIDs/',
    'Path to save train id files.')

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    # Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  20,  20,  20) ), # in gta dataset it has a different color
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


# annotation_folder = '/home/mumu01/Downloads/datasets/gta/labels/'

def create_folders(fol_name):
    if not (os.path.exists(fol_name)):
        os.makedirs(fol_name)


annotation_folder = FLAGS.gta_labels
trainID_folder = FLAGS.trainID_dir

create_folders(trainID_folder)

considered_labels = [label for label in labels if not (label.ignoreInEval)]
not_considered_labels = [label for label in labels if (label.ignoreInEval)]


def get_file_names(dir_name):
    only_files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    only_files.sort()
    return only_files


def get_gtFine_labelTrainIds(color_annotated_img):
    # create similar size image as annotation

    r, c, _ = np.shape(color_annotated_img)
    labelTrainIds_img = np.zeros((r, c), dtype=np.uint8)

    # handle objects which needs to be considered by giving them same id as Train Id
    for l in considered_labels:
        lookup_color = l.color
        lookup_color = (
            lookup_color[2], lookup_color[1],
            lookup_color[0])  # switching blue and red channel of color to synchronize.

        lower = np.array(lookup_color, dtype="uint8")
        upper = np.array(lookup_color, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(color_annotated_img, lower, upper)
        labelTrainIds_img[mask != 0] = l.trainId

    # handle objects which are not getting considered and assign them a different id
    for l in not_considered_labels:
        if not (l.name == 'license plate') and not (l.name == 'polegroup'):
            lookup_color = l.color
            lookup_color = (
                lookup_color[2], lookup_color[1],
                lookup_color[0])  # switching blue and red channel of color to synchronize.

            lower = np.array(lookup_color, dtype="uint8")
            upper = np.array(lookup_color, dtype="uint8")

            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(color_annotated_img, lower, upper)
            labelTrainIds_img[mask != 0] = 255  # trainID for unlabeled

    return labelTrainIds_img


# depreciated
def get_colored_img_from_gtFine_labelTrainIds(labelTrainIds_img):
    r, c = np.shape(labelTrainIds_img)
    colored_img = np.zeros((r, c, 3), dtype=np.uint8), 255
    for l in labels:
        if not (l.trainId == 255) and not ((l.trainId == -1)):
            lookup_trainID = l.trainId
            lower = np.array(lookup_trainID, dtype="uint8")
            upper = np.array(lookup_trainID, dtype="uint8")

            mask = cv2.inRange(labelTrainIds_img, lower, upper)

            assign_color = l.color
            assign_color = (
                assign_color[2], assign_color[1],
                assign_color[0])  # switching blue and red channel of color to synchronize.

            colored_img[np.where(mask == [255])] = assign_color

    return colored_img



file_names = get_file_names(annotation_folder)
# file_names = ['00002.png', '00003.png', '00005.png']
num_files = len(file_names)
sample_annotation_files = file_names[:num_files]

for s in sample_annotation_files:
    full_path = join(annotation_folder, s)
    color_annotated_img = cv2.imread(full_path)

    labelTrainIds_img = get_gtFine_labelTrainIds(color_annotated_img)
    cv2.imwrite(join(trainID_folder, s),
                labelTrainIds_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # need to make sure if saving doesn't effect the values in side

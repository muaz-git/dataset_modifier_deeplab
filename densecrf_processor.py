import os
from os import listdir
from os.path import isfile, join, isdir
from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='Script to apply DenseCRF to annotations (colored).')

parser.add_argument('-e', '--exp_vis', dest='exp_vis', action="store",
                    help="Path to the directory which is generated in the result of deeplap/vis.py of an experiment.",
                    required=True)

parser.add_argument('-n', '--num', dest='exp_num', action="store", help="number of an experiment", type=int, required=True)
results = parser.parse_args()

densecrf_loc = '/home/mumu01/scripts/densecrf/build/examples/dense_inference'
if not (isfile(densecrf_loc)):
    raise ValueError('Script of Dense CRF not found in location : ' + densecrf_loc)

if not (isdir(results.exp_vis)):
    raise ValueError('Directory doesn\'t exist : ' + results.exp_vis)

source_imgs_loc = results.exp_vis + '/images'
source_annot_loc = results.exp_vis + '/colored'

if not (isdir(source_imgs_loc)):
    raise ValueError('Directory doesn\'t exist : ' + source_imgs_loc)

if not (isdir(source_annot_loc)):
    raise ValueError('Directory doesn\'t exist : ' + source_annot_loc)

crf_file_loc = results.exp_vis + '/crf_processed'

if not os.path.exists(crf_file_loc):
    os.makedirs(crf_file_loc)

image_list = [f for f in listdir(source_imgs_loc) if isfile(join(source_imgs_loc, f))]
annotation_list = [f for f in listdir(source_annot_loc) if isfile(join(source_annot_loc, f))]

image_list.sort()
annotation_list.sort()

exp_num = int(results.exp_num)

for i, a in zip(image_list, annotation_list):
    if not i == a:
        raise ValueError('Names of file mismatch')
    print(i)

    img = Image.open(source_imgs_loc + '/' + i)
    img.save('tmp_'+str(exp_num)+'.ppm')

    annot = Image.open(source_annot_loc + '/' + a)
    annot.save('tmp_annot_'+str(exp_num)+'.ppm')
    command = densecrf_loc + ' tmp_'+str(exp_num)+'.ppm tmp_annot_'+str(exp_num)+'.ppm' + ' ' + crf_file_loc + '/' + i[:-4] + '.ppm'
    # print(command)
    os.system(command)
    processed = Image.open(crf_file_loc + '/' + i[:-4] + '.ppm')
    processed.save(crf_file_loc + '/' + i[:-4] + '.png')
    os.remove(crf_file_loc + '/' + i[:-4] + '.ppm')
    # break

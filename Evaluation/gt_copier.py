from os import listdir
from os.path import isfile, join
import os
from shutil import copyfile

dataset_dir = './datasets'


def get_file_names(dir_name):
    only_files = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    only_files.sort()
    return only_files


def copy_gta():
    gta_dir = dataset_dir + '/gta'
    pred_gta = gta_dir + '/predictions'

    gt_source_dir_gta = '/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01/home/mumu01/models/deeplab/datasets/gta/Ids'
    gt_dst_dir_gta = gta_dir + '/groundtruth'
    filenames = get_file_names(pred_gta)
    for f in filenames:
        if not isfile(gt_source_dir_gta + '/' + f):
            print(gt_source_dir_gta + '/' + f, ' doesnt exist')

        else:
            copyfile(gt_source_dir_gta + '/' + f, gt_dst_dir_gta + '/' + f)

def copy_cityscapes():
    citscapes_dir = dataset_dir + '/cityscapes'
    pred_cs = citscapes_dir + '/predictions'
    gt_source_dir_cs= '/run/user/477015036/gvfs/sftp:host=sonic.sb.dfki.de,user=mumu01/home/mumu01/models/deeplab/datasets/cityscapes'
    gt_dst_dir_cs = citscapes_dir + '/groundtruth'

    import glob

    groundTruthSearch = os.path.join(gt_source_dir_cs, "gtFine", "val", "*", "*_gtFine_labelIds.png")
    groundTruthImgList = glob.glob(groundTruthSearch)

    for comp_path in groundTruthImgList:
        f = os.path.basename(comp_path)
        if not isfile(comp_path):
            print(comp_path, ' doesnt exist')

        else:
            copyfile(comp_path, gt_dst_dir_cs + '/' + f)

# citscapes_dir = dataset_dir + '/cityscapes'
# pred_cs = citscapes_dir + '/predictions'
# gt_dst_dir_cs = citscapes_dir + '/groundtruth'
# print(len(get_file_names(pred_cs)))
# print(len(get_file_names(gt_dst_dir_cs)))

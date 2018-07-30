# Dataset_modifier_deeplab
This repository includes code which converts datasets i.e. Playing for data, into `tfrecords` so that it will be acceptable by [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) project.

## Where to place the files?
Copy the content of this repository in to folder: `tensorflow/models/research/deeplab/datasets`, except `densecrf_processor.py` and `vis.py`.

## Steps to do beforehand
Download data i.e. from [here](https://download.visinf.tu-darmstadt.de/data/from_games/).
Extract them in `tensorflow/models/research/deeplab/datasets` such that you have following directory structure.
```
+ datasets
  + gta
    + images
    + labels
```


## How to run?
```bash
# From the tensorflow/models/research/deeplab/datasets directory.
sh convert_gta_pfd.sh.sh
```

# DenseCRF
In order to run DenseCRF, run `python densecrf_processor.py -e /home/mumu01/exps/exp42/vis/ -n 42`. Where `-e` contains the path to the directory of the results of `vis.py` and `-n` contains a different number so that `ppm` files dont conflict incase of multithreading of this script.

# Visualization
`vis.py` is a bit modified version of original `deeplab/vis.py`.

## Changes include:
```
+ creates new folders.
+ breaks the loop in order to check for more checkpoint files.
```

## Where to place `vis.py`
Place `vis.py` in `deeplab/` directory.

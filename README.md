# Dataset_modifier_deeplab
This repository includes code which converts datasets i.e. Playing for data, into `tfrecords` so that it will be acceptable by [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) project.

## Where to place the files?
Copy the content of this repository in to folder: `tensorflow/models/research/deeplab/datasets`

## Steps to do beforehand
Download data from [here](https://download.visinf.tu-darmstadt.de/data/from_games/).
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

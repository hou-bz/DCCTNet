# Structure of this repository
This repository is organized as:

- [datasets](/datasets/) contains the dataloader for different datasets
- [networks](/networks/) contains a model zoo for network models
- [scripts](/networks/) coontains scripts for preparing data
- [utils](/networks/) contains api for training and processing data
- [train.py](/train.py) train a single model

# Usage Guide

## Requirements

 All the codes are tested in the following environment:

- pytorch 1.8.0
- pytorch-lightning >= 1.3.7
- OpenCV
- nibabel

## Dataset Preparation

### KiTS
Download data [here](https://github.com/neheller/kits19)

Please follow the instructions and the data/ directory should then be structured as follows
```
data
├── case_00000
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
├── case_00001
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
...
├── case_00209
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
└── kits.json
```
Cut 3D data into slices using ```scripts/SliceMaker.py``` 

```
python scripts/SliceMaker.py --inpath /data/kits19/data --outpath /data/kits/train --dataset kits --task tumor
```

```
python scripts/SliceMaker.py --inpath /data/lits/Training-Batch --outpath /data/lits/train --dataset lits --task tumor
```

## Running
### Training 
```
python train.py --model unet --checkpoint_path /data/checkpoints
```

After training, the checkpoints will be stored in ```/data/checkpoints``` as assigned.

If you want to try different models, use ```--model``` with following choices
```
'deeplabv3+', 'enet', 'erfnet', 'espnet', 'mobilenetv2', 'unet++', 'raunet', 'resnet18', 'unet', 'pspnet'
```
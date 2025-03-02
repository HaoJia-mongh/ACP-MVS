# ACP-MVS
This repository contains the source code for our paper:

ACP-MVS: Efficient Multi-View Stereo with Attention-based Context Perception

## Environment
* NVIDIA RTX 3090
* python 3.8
* CUDA >= 11.1
```
pip install -r requirements.txt
```

## Required Data
To evaluate/train ACP-MVS, you will need to download the required datasets. 

*[DTU's evaluation](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing)

*[DTU's training](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)

*[Tanks and Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view?usp=sharing)

*[BlendedMVS](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV)


## Reproducing Results
* Download pre-processed datasets (provided by PatchmatchNet): [DTU's evaluation set](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing), [Tanks & Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view?usp=sharing)
```
root_directory
├──scan1 (scene_name1)
├──scan2 (scene_name2) 
      ├── images                 
      │   ├── 00000000.jpg       
      │   ├── 00000001.jpg       
      │   └── ...                
      ├── cams_1                   
      │   ├── 00000000_cam.txt   
      │   ├── 00000001_cam.txt   
      │   └── ...                
      └── pair.txt  
```

Camera file ``cam.txt`` stores the camera parameters, which includes extrinsic, intrinsic, minimum depth and maximum depth:
```
extrinsic
E00 E01 E02 E03
E10 E11 E12 E13
E20 E21 E22 E23
E30 E31 E32 E33

intrinsic
K00 K01 K02
K10 K11 K12
K20 K21 K22

DEPTH_MIN DEPTH_MAX 
```
``pair.txt `` stores the view selection result. For each reference image, 10 best source views are stored in the file:
```
TOTAL_IMAGE_NUM
IMAGE_ID0                       # index of reference image 0 
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 0 
IMAGE_ID1                       # index of reference image 1
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 1 
...
``` 


### Evaluation on DTU:
* In ``test.sh``, set `DTU_TESTING`, or `TANK_TESTING` as the root directory of corresponding dataset, set `--OUT_DIR` as the directory to store the reconstructed point clouds, uncomment the evaluation command for corresponding dataset (default is to evaluate on DTU's evaluation set).
* `CKPT_FILE` is the checkpoint file (our pretrained model is `checkpoints/DTU.ckpt`). 
* Test on GPU by running `sh test.sh`. The code includes depth map estimation and depth fusion. The outputs are the point clouds in `ply` format. 
* For quantitative evaluation, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36) from DTU's website. Unzip them and place `Points` folder in `SampleSet/MVS Data/`. The structure looks like:
```
SampleSet
├──MVS Data
      └──Points
```
In ``evaluations/dtu/BaseEvalMain_web.m``, set `dataPath` as the path to `SampleSet/MVS Data/`, `plyPath` as directory that stores the reconstructed point clouds and `resultsPath` as directory to store the evaluation results. Then run ``evaluations/dtu/BaseEvalMain_web.m`` in matlab.

The results look like:


| Acc. (mm) | Comp. (mm) | Overall (mm) |
|-----------|------------|--------------|
| 0.315     | 0.285      | 0.300        |


### Evaluation on Tansk and Temples:
* In ``test.sh``, set `TANK_TESTING` as the root directory of the dataset and `--outdir` as the directory to store the reconstructed point clouds. 
* `CKPT_FILE` is the path of checkpoint file (our pretrained model is `checkpoints/TANK_train_on_dtu.ckpt`). We also provide our pretrained model trained on BlendedMVS (`checkpoints/TANK_train_on_blendedmvs.ckpt`)
* Test on GPU by running `sh test.sh`. The code includes depth map estimation and depth fusion. The outputs are the point clouds in `ply` format. 


TANK on DTU(mean F-score)
| intermediate | advanced (mm) |
|------------- |-------------- |
| 59.81        | 37.41         |


TANK on blendedmvs(mean F-score)
| intermediate | advanced (mm) |
|------------- |-------------- |
| 64.70        | 41.72         |


## Training


### DTU
* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) 
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and upzip it as the $MVS_TRANING  folder.
* In ``train.sh``, set `MVS_TRAINING` as the root directory of dataset; set `--logdir` as the directory to store the checkpoints. 
* Train the model by running `sh train.sh`.


### BlendedMVS
* Download the [BlendedMVS dataset](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV).
* In ``train.sh``, set `MVS_TRAINING` as the root directory of dataset; set `--logdir` as the directory to store the checkpoints. 
* Train the model by running `sh train.sh`.



# Acknowledgements
This project is based on [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch), [Effi-MVS](https://github.com/bdwsq1996/Effi-MVS). We thank the original authors for their excellent works.


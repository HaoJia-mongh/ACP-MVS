# ACP-MVS  

This repository contains the source code for our paper:  

**ACP-MVS: Efficient Multi-View Stereo with Attention-based Context Perception**  

---

## Environment  
- **Hardware**: NVIDIA RTX 3090  
- **Python**: 3.8  
- **CUDA**: >= 11.1  

To install dependencies, run:  
```bash
pip install -r requirements.txt
```

## **Required Data**  
To evaluate or train ACP-MVS, you need to download the required datasets:  

- [DTU Evaluation Set](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing)  
- [DTU Training Set](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)  
- [Tanks and Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view?usp=sharing)  
- [BlendedMVS](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV)  


## **Reproducing Results**  

### **Dataset Structure**  
Download the preprocessed datasets (provided by PatchmatchNet):  

- [DTU Evaluation Set](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing)  
- [Tanks and Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view?usp=sharing)  

The dataset directory structure should be as follows:  
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

## **File Descriptions**  

### **Camera File (`cam.txt`)**  
Stores camera parameters, including extrinsics, intrinsics, minimum depth, and maximum depth:
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

### **View Selection File (`pair.txt`)**  
Stores the best 10 source views for each reference image:  

```
TOTAL_IMAGE_NUM
IMAGE_ID0                       # Reference image 0 index
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 0 
IMAGE_ID1                       # Reference image 1 index
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 1 
...
``` 

## **Evaluation**  

### **Evaluation on DTU**  
1. In `test.sh`, set `DTU_TESTING` as the root directory of the corresponding dataset.  
2. Set `--OUT_DIR` as the directory to store the reconstructed point clouds.  
3. Uncomment the evaluation command for the desired dataset (default is DTU).  
4. Set `CKPT_FILE` to the checkpoint file (`checkpoints/DTU.ckpt` for our pretrained model).  
5. Run the evaluation on GPU using:  
```bash
sh test.sh
```
The code will generate depth maps and perform depth fusion. The outputs are point clouds in `.ply` format.

6. For quantitative evaluation, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36) from the DTU website. Extract the `Points` folder into `SampleSet/MVS Data/`, ensuring the structure is:
```plaintext
SampleSet
├── MVS Data
    └── Points
```
7. In `evaluations/dtu/BaseEvalMain_web.m`, set:

- `dataPath` → path to `SampleSet/MVS Data/`
- `plyPath` → directory storing the reconstructed point clouds
- `resultsPath` → directory to store evaluation results

8. Run `evaluations/dtu/BaseEvalMain_web.m` in MATLAB.

### **Evaluation Results (DTU)**

| Accuracy (mm) | Completeness (mm) | Overall (mm) |
|---------------|-------------------|--------------|
| 0.315         | 0.285             | 0.300        |



### **Evaluation on Tanks and Temples**

1. In `test.sh`, set `TANK_TESTING` as the dataset root directory.  
2. Set `--outdir` as the directory for the reconstructed point clouds.  
3. Specify the checkpoint file:  
   - `checkpoints/TANK_train_on_dtu.ckpt` (trained on DTU)  
   - `checkpoints/TANK_train_on_blendedmvs.ckpt` (trained on BlendedMVS)  
4. Run the evaluation on GPU using:  
```bash
sh test.sh
```
The code will generate depth maps and perform depth fusion. The outputs are point clouds in `.ply` format.


### **Evaluation Results**

#### **DTU-trained Model (Mean F-score)**

| Level        | Intermediate | Advanced |
|--------------|--------------|----------|
| **F-score**  | 59.81        | 37.41    |

#### **BlendedMVS-trained Model (Mean F-score)**

| Level        | Intermediate | Advanced |
|--------------|--------------|----------|
| **F-score**  | 64.70        | 41.72    |


### **Training**

#### **Training on DTU**

1. Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) 
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)).
2. Extract them into `$MVS_TRAINING`.
3. In `train.sh`, set `MVS_TRAINING` as the dataset root directory.
4. Set `--logdir` as the directory to store checkpoints.
5. Train the model by running:
```bash
sh train.sh
```


#### **Training on BlendedMVS**

1. Download the [BlendedMVS dataset](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV).
2. In `train.sh`, set `MVS_TRAINING` as the dataset root directory.
3. Set `--logdir` as the directory to store checkpoints.
4. Train the model by running:

```bash
sh train.sh
```

### **Acknowledgements**

This project is based on:

- [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch)
- [Effi-MVS](https://github.com/bdwsq1996/Effi-MVS)

We thank the original authors for their excellent work.


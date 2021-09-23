# CMR_plan
This repository houses codes for automatic standard view planning for cardiac MR (CMR) imaging from a series of localizers via self-supervision by spatial relationship between views, in accordance with our MICCAI 2021 paper.

## Introduction
We provide an example data (case) located in the path "./data", which is used as both train and valiadtion (test) data for code demonstration purpose.

## Step 1: Preprocess localizers
The first step is to prepare the localizers based on which the standard views are prescribed, from raw DICOM files
``` 
cd prep_data
python prepare_2C_loc.py
python prepare_4C_loc.py
python prepare_SAX_loc.py
cd ..
```
The prepared data are saved in four folders named in the pattern "view_plan_data_xxx", in path "./data".

## Step 2: Train regression networks
Next is to train view-specific regression networks for each localizer:
``` 
python train.py --view 2C
python train.py --view 4C
python train.py --view SAX
```
After training, the models are saved in "./models". We provide pretrained models as described in our paper, together with the code.

## Step 3: Predict heatmaps with trained networks
Now, we can predict view-planning heatmaps for test data with the trained networks:
``` 
python inference.py --view 2C --prev models\2C_HT0.5.pth.tar
python inference.py --view 4C --prev models\4C_HT0.5.pth.tar
python inference.py --view SAX --prev models\SAX_HT0.5.pth.tar
```
The prediction results are saved in three folders named in the pattern "view_plan_pred_xxx_loc", in path "./data".

## Step 4: Prescribe standard views by multi-view aggregation
The last step is to prescribe standard views by multi-view aggregation of the heatmap prediction results:
``` 
python prescribe_std_views.py
```
The prescribed planes are presented as ImagePositionPatient (IPP) and ImageNormalPatient (INP); please refer to the codes. For inituitive perception, we visualize the prescription results in the folder "prescribe_std_views" located in path "./data", where green color indicates the ground truth, red color indicates the automatic prescription, and yellow color indicates their overlap.

## Acknowledgement
Kindly cite us if you use our codes in your projects:
- D. Wei, K. Ma, and Y. Zheng, "Training Automatic View Planner for Cardiac MR Imaging via Self-Supervision by Spatial Relationship between Views," accepted by  *International Conference on Medical Image Computing and Computer-Assisted Intervention* (MICCAI 2021).

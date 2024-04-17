### Automation Lab, Sungkyunkwan University

# ETSS-07: Traffic Congestion Detection

This is the official repository of 

**MetaCrowd: A Unified Framework for Crowd Counting and Traffic Congestion Detection.**

## Setup
```bash
conda env create -f environment.yml
conda activate anomaly
```

## Dataset Preparation
For RGBT-CC dataset, please download it from this [link](https://lingboliu.com/RGBT_Crowd_Counting.html).

For ShanghaiTech RGB-D dataset, please download it from this [repo](https://github.com/svip-lab/RGBD-Counting).

## Usage
To use our model, follow the code snippet bellow:
```bash
# Train and Test CSCA model
bash train_rgbt_cc.sh
bash test_rgbt_cc.sh
bash train_shanghai_rgbd.sh
bash test_shanghai_rgbd.sh
```

#### Supported models
------------------------------------------------------
| Models           | RGBT-CC            | ShanghaiTech RGB-D |
|------------------|--------------------|--------------------|
| CSCA (ACCV 2022) | :heavy_check_mark: | :heavy_check_mark: |

## Citation
If you find our work useful, please cite the following:
```
@misc{Chi2023,
  author       = {Chi Tran},
  title        = {MetaCrowd: A Unified Framework for Crowd Counting and Traffic Congestion Detection},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/SKKU-AutoLab-VSW/ETSS-07-CongestionDetection},
  year         = {2024}
}
```

## Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com)).

##  Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.
<!--ts-->
* [chen-judge/RGBTCrowdCounting](https://github.com/chen-judge/RGBTCrowdCounting)
* [AIM-SKKU/CSCA](https://github.com/AIM-SKKU/CSCA)
* [ZhihengCV/Bayesian-Crowd-Counting](https://github.com/ZhihengCV/Bayesian-Crowd-Counting)
<!--te-->
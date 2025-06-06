# CDSegNet

This repo is the official project repository of the paper **_An End-to-End Robust Point Cloud Semantic Segmentation Network with Single-Step Conditional Diffusion Models_**. 
 -  [ [arXiv](https://arxiv.org/abs/2411.16308) ], [ [CVPR](https://openaccess.thecvf.com/content/CVPR2025/papers/Qu_An_End-to-End_Robust_Point_Cloud_Semantic_Segmentation_Network_with_Single-Step_CVPR_2025_paper.pdf) ] (The CVPR version has slightly blurred images due to transcoding issues, so it is recommended to check the arXiv version.)
 - **_Our paper has been accepted by CVPR 2025!_**
 - **_Released model weights are temporarily as the model structure of CDSegNet may be adjusted later._**
## The Overall Framework 
<img src="assets/overall.png" alt="cdsegnet" width="900"/> <br/>
**_CN : Conditional Network_** <br/> **_NN : Noise Network_** <br/>
1) For end-to-end traditional DDPMs (Noise-Contional Framework, NCF), the Conditional Network (CN) extracts the conditional features for generating guidance. Meanwhile, the Noise Network (NN) predicts the scores from the task target, dominating the results of tasks (see left (a)).
2) For CNF (Conditional-Noise Framework), CNF treats NN and CN as the auxiliary network and the dominant network in 3D tasks (see left (b)), respectively.
## Citation
If you find our paper useful to your research, please cite our work as an acknowledgment.
```bib
@inproceedings{qu2025end,
  title={An end-to-end robust point cloud semantic segmentation network with single-step conditional diffusion models},
  author={Qu, Wentao and Wang, Jing and Gong, YongShun and Huang, Xiaoshui and Xiao, Liang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={27325--27335},
  year={2025}
}
```

## Motivation
Below, we give a brief explanation of the motivation of our paper, hoping to help readers further understand our idea.

#### Traditional DDPMs excel in generative tasks but are limited to 3D perception tasks due to **_multi-step iterations_** and **_the difficulty fitting target scores (the gredients of target distribution, noise &epsilon; or target x0)_**. 
 - 1) **_multi-step iterations:_** PTv3 infers on ScanNet in 22s (no test-time augmentation (TTA), performing inference for four 4090 NVIDIA GPUs), while a DDPM built on PTv3 takes nearly 7h for 1000 steps (Fig.2). Even with acceleration strategies, but: 1) suboptimal results, 2) still multi-step requirements.
 - 2) **_the difficulty of fitting semantic label scores:_** It is unrealistic to directly fit the score of point cloud semantic labels according to traditional DDPMs in an end-to-end manner, because the distribution of 3D scenes is too complex than that of 2D scenes (the best proof is that DDPMs can be used to achieve image segmentation in an end-to-end manner, but so far, there is no successful case for point cloud segmentation). We believe that this is similar to using DDPMs to do 3D scene generation tasks in an end-to-end manner. This is extremely difficult. **_For example, non-DDPMs models only need to fit semantic labels y given point cloud c, but DDPMs require that conditional point cloud c and x={x0,x1,...,xt} (semantic labels with different levels of noise) be given, and fit y={y0,y1,...,yt} (scores of different levels, noise &epsilon; or semantic label x0). This similarly increases the difficult degree for fitting the network by t times._** 
#### So why do we still use DDPMs for semantic segmentation tasks? Isn’t it better for us to use non-DDPMs directly?
 - This is the main contribution of our paper. 
 - 1) **_Our paper analyzes the advantages (noise and sparsity robustness) and weaknesses (more iterations) of DDPMs in 3D perception tasks._** 
 - 2) **_We attempt to separate the advantages and limitations of DDPMs, and as a result, we propose CNF, a novel end-to-end framework of DDPMs that maintains performance and robustness while avoiding iterations. The key idea is to use CN as the dominant network, determining the segmentation result, and NN as the auxiliary network, enhancing the features in CN._**
#### Why can our CNF effectively maintains advantages while avoiding limitations of DDPMs?
 - 1) **_As CN is as the dominant network, avoiding multiple iterations during inference, because the output from NN is no longer important._**
 - 2) **_As CN is as the dominant network, alleviate the necessity of excessively fitting the scores from the task target for NN, improving the convergence speed of training._**
 - 3) **_Moveover, CNF maintains the DDPMs training rules during training, thus preserving data sparsity and noise robustness._**
 - **_Models with CNF still essentially qualify as DDPMs, as trained models can perform DDPM inference (CNF follows the DDPMs training rules). The key difference from traditional DDPMs is simply the output from CN instead of NN in inference._**
 - As a framework, CNF is built on a backbone (e.g.~PTv3),  thus inevitably slightly increasing computational cost. However, CNF is more cost-effective compared to traditional DDPMs (NCF) (see Fig.2). 
#### Our paper's aim is not to propose a novel backbone but to **_introduce a new perspective for applying DDPMs to 3D perception tasks_** (CDSegNet is an instance of CNF).

## Explanation of Figure 2
<img src="assets/combinations.png" alt="combinations" width="900"/> <br/>
<img src="assets/compare.png" alt="compare" width="900"/> <br/>

- We try several combinations for conditional DDPMs built on the baseline (②) on ScanNet in (a). The same architecture as CDSegNet(CNF,①) but using traditional DDPMs for segmentation (NCF,③,④,⑤,⑥)
- (b) shows the inference time cost of CNF and NCF under the same baseline. Due to multiple iterations, the reasoning time of NCF is even close to 7 hours.
- ③ : CN+GD -> This auxiliary network does not have a diffusion process (regressing the point cloud coordinates via MSE), while the dominant network has a [[Gaussion](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)] diffusion process. The result is dominated the NN.
- ④ : GD+GD -> This auxiliary network have a [[Gaussion](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)] diffusion process (the input contains random noise and conditional point cloud), while the dominant network has a [[Gaussion](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)] diffusion process. The result is dominated the NN.
- ⑤ : CN+CD -> This auxiliary network does not have a diffusion process (regressing the point cloud coordinates via MSE), while the dominant network has a [[Categorical](https://proceedings.neurips.cc/paper/2021/file/958c530554f78bcd8e97125b70e6973d-Paper.pdf)] diffusion process. The result is dominated the NN.
- ⑥ : GD+CD -> This auxiliary network have a [[Gaussion](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)] diffusion process (the input contains random noise and conditional point cloud), while the dominant network has a [[Categorical](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)] diffusion process. The result is dominated the NN.
- Note: **_The above combination (NCF,③,④,⑤,⑥) is the same as the network framework of Baseline (②) and CDSegNet (CNF,①)._**
- **_For the implementation of these combinations, please see [[default.py](https://github.com/QWTforGithub/CDSegNet/blob/main/pointcept/models/default.py)]_**

## Overview
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Zoo](#model-zoo)
- [Quick Start](#quick-start)

## Installation

### Requirements
The following environment is recommended for running **_CDSegNet_** (an NVIDIA 3090 GPU or four NVIDIA 4090 GPUs):
- Ubuntu: 18.04 and above
- gcc/g++: 11.4 and above
- CUDA: 11.8 and above
- PyTorch: 2.1.0 and above
- python: 3.8 and above

### Environment

- Base environment
```
sudo apt-get install libsparsehash-dev

conda create -n cnf python=3.8 -y
conda activate cnf
conda install ninja -y

conda install google-sparsehash -c bioconda

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric
pip install spconv-cu118
pip install open3d

# compile C++ extension packages
# Please ensure:
#   nvcc : 11.8
#   gcc/g++ : 11.4 
cd CDSegNet-main/scripts
sh compile.sh

# install flash-attention
# 1. cuda11.8 -> cuda11.6
#   vim ~/.bashrc
#   export PATH="/usr/local/cuda-11.8/bin:$PATH" -> export PATH="/usr/local/cuda-11.6/bin:$PATH"
#   export CUDA_HOME="/usr/local/cuda-11.8" -> export CUDA_HOME="/usr/local/cuda-11.6"
# 2. please download flash_attn-2.5.7+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl from:
#      a. The official website: https://github.com/Dao-AILab/flash-attention/releases?page=2
#      b. Our links: Baidu Netdisk and Google Drive in Model Zoom
# 3. pip install flash_attn-2.5.7+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## Data Preparation
- Please refer to the [data processing](https://github.com/Pointcept/Pointcept#data-preparation) of PTv3.
### ScanNet/ScanNet200
- The preprocess data of PTv3 can be directly downloaded [[here](https://huggingface.co/datasets/Pointcept/scannet-compressed)], please agree the official license before download it.

- Link processed dataset to codebase:
  ```bash
  # PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset.
  mkdir data
  ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
  ```

### nuScenes
- Download the official [nuScenes](https://www.nuscenes.org/nuscenes#download) dataset (with Lidar Segmentation) and organize the downloaded files as follows:
  ```bash
  NUSCENES_DIR
  │── samples
  │── sweeps
  │── lidarseg
  ...
  │── v1.0-trainval 
  │── v1.0-test
  ```
- The preprocess nuScenes information data can also be downloaded [[here](https://huggingface.co/datasets/Pointcept/nuscenes-compressed)] (only processed information, still need to download raw dataset and link to the folder), please agree the official license before download it.

- Link raw dataset to processed NuScene dataset folder:
  ```bash
  # NUSCENES_DIR: the directory of downloaded nuScenes dataset.
  # PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
  ln -s ${NUSCENES_DIR} {PROCESSED_NUSCENES_DIR}/raw
  ```
  then the processed nuscenes folder is organized as follows:
  ```bash
  nuscene
  |── raw
      │── samples
      │── sweeps
      │── lidarseg
      ...
      │── v1.0-trainval
      │── v1.0-test
  |── info
  ```

- Link processed dataset to codebase.
  ```bash
  # PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
  mkdir data
  ln -s ${PROCESSED_NUSCENES_DIR} ${CODEBASE_DIR}/data/nuscenes
  ```

## Model Zoo
<img src="assets/compare_table.png" alt="compare_table" width="900"/> 

### Indoor Benchmark
| Model | Benchmark | Only Training Data? | Num GPUs | Val mIoU | checkpoint |
| :---: | :---: |:---------------:| :---: | :---: | :---: |
| CDSegNet | ScanNet |     &check;     | 1,2,4 | 77.9% | [Link1](https://pan.baidu.com/s/1rAUHa4OHmT_Q1I2Pi_sVog?pwd=1111), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 + CNF | ScanNet |     &check;     | 4 | 77.7% | [Link1](https://pan.baidu.com/s/1rAUHa4OHmT_Q1I2Pi_sVog?pwd=1111), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 | ScanNet |     &check;     | 4 | 77.6% | [Link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/scannet-semseg-pt-v3m1-0-base/model) |
| CDSegNet | ScanNet200 |     &check;     | 4 | 36.3% | [Link1](https://pan.baidu.com/s/1rAUHa4OHmT_Q1I2Pi_sVog?pwd=1111), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 + CNF | ScanNet200 | &check;  | 4 | 35.9% | [Link1](https://pan.baidu.com/s/1rAUHa4OHmT_Q1I2Pi_sVog?pwd=1111), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 | ScanNet200 | &check;  | 4 | 35.3% | [Link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/scannet200-semseg-pt-v3m1-0-base/model) |

### Outdoor Benckmark
| Model | Benchmark | Only Training Data? | Num GPUs | Val mIoU | checkpoint |
| :---: | :---: |:---------------:| :---: | :---: | :---: |
| CDSegNet | nuScenes |     &check;     | 4 | 81.2% | [Link1](https://pan.baidu.com/s/1rAUHa4OHmT_Q1I2Pi_sVog?pwd=1111), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 + CNF | nuScenes |     &check;     | 4 | 81.0% | [Link1](https://pan.baidu.com/s/1rAUHa4OHmT_Q1I2Pi_sVog?pwd=1111), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 | nuScenes |     &check;     | 4 | 80.3% | [Link1](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/nuscenes-semseg-pt-v3m1-0-base/model) |


## Quick Start

### Training
We provide two indoor datasets (ScanNet, ScanNet200) and one ourdoor dataset (nuScenes) to train CDSegNet. The results are in the 'CDSegNet-main/exp/{dataset}/{config}' folder.
GPUs and batch size are not limited. We successfully generate 77.9% mIoU on ScanNet with **_1 (BS=2), 2 (BS=4), and 4 (BS=8) GPUs [Link](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing)_**.
```
#  Configure the dataset path:
#    a. CDSegNet-main/configs/{dataset}/CDSegNet.py
#    b. data_root = "Your dataset path", for example, data_root = "/../CDSegNet-main/data/scannet or nuscenes"
# Training on ScanNet
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_CDSegNet_ScanNet.py
# Training on ScanNet200
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_CDSegNet_ScanNet200.py

# Training on nuScenes
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_CDSegNet_nuScenes.py
```

#### Important!!! : Training Tricks
- CDSegNet is tied to PTv3, but we found that **_the training of PTv3 is unstable_**, and even with a fixed seed we cannot ensure that the results are roughly the same each time (fluctuations are around 1.0%mIoU).
- The reasons for instability are: 1) Grid pooling 2) Sparse convolution
- This once caused us headaches and sadness, because when we adjusted the parameters, we could not determine whether the poor performance was due to parameter problems or randomness problems.
- With unremitting efforts, we found a way to stabilize performance. **_That is to save the checkpoint in the middle, and then load the training repeatedly_**. For example, on ScanNet, 100 epochs are required, and we save the checkpoint at 70 epoch. Then, repeatedly train from 70 epoch to 100 epoch. This may get the most stable results.

#### Extensions
- If you want to extend our work, we recommend using PTv3+CNF instead of CDSegNet. Since PTv3+CNF has only half the parameters of CDSegNet, the performance of the two is quite close.
- We found that **_on the nuScene testing set_**, **_[[PTv3+CNF](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing)] achieved 82.8% mIoU (PTv3 ->81.2%, CDSegNet->82.0% mIoU) when trained only on the training set of nuScenes._** PTv3+PPT showed 83.0% mIoU, but PTv3+PPT has the double number of parameters than PTv3+CNF and uses multiple datasets for joint training (we guess there are even 5 datasets).
- Core files:
- 1)  [[default.py](https://github.com/QWTforGithub/CDSegNet/blob/main/pointcept/models/default.py)] : Implementation of different DDPMs (NCF and CNF).
  2)  [[point_transformer_v3m1_base.py](https://github.com/QWTforGithub/CDSegNet/blob/main/pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py)] : Implementation of the backbone.
  3)  [[train.py](https://github.com/QWTforGithub/CDSegNet/blob/main/pointcept/engines/train.py)] : Implementation of the training processing (Trainer).
  4)  [[test.py](https://github.com/QWTforGithub/CDSegNet/blob/main/pointcept/engines/test.py)] : Implementation of the testing processing (SemSegTester).
  5)  [[evaluator.py](https://github.com/QWTforGithub/CDSegNet/blob/main/pointcept/engines/hooks/evaluator.py)] : Implementation of the validating processing (SemSegEvaluator).
  6)  [[scannet.py](https://github.com/QWTforGithub/CDSegNet/blob/main/pointcept/datasets/scannet.py)] : Implementation of the scannet dataloader (get_data_list(), get_data()).
  7)  [[nuscenes.py](https://github.com/QWTforGithub/CDSegNet/blob/main/pointcept/datasets/nuscenes.py)] : Implementation of the nuscenes dataloader (get_data_list(), get_data()).

### Experiments of Sparsity Robustness
-  This first randomly samples 5%, 10%, 12.5%, 25%, and 50% from the training and validation set, respectively.
-  Subsequently, the model is trained and fitted on the under-sampled training and validation set, while performing inference on the entire validation set.

### Testing
We provide two indoor datasets (ScanNet, ScanNet200) and one ourdoor dataset (nuScenes) to test CDSegNet. The results are in the 'CDSegNet-main/exp/{dataset}_test/{config}' folder.
```
#  1. Configure the dataset path:
#    a. CDSegNet-main/configs/{dataset}/CDSegNet.py
#    b. data_root = "Your dataset path", for example, data_root = "/../CDSegNet-main/data/scannet or nuscenes"
#  2. Configure the weight path:
#    a. CDSegNet-main/tools/train_CDSegNet_{dataset}.py
#    b. weight = "the downloading checkpoint"

# Testing on ScanNet
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_CDSegNet_ScanNet.py
# We also can test at a specified noise level on ScanNet:
# CDSegNet-main/tools/train_CDSegNet_{dataset}.py, noise_level=0.1 (0.01~0.1)
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_CDSegNet_ScanNet.py
# Testing on ScanNet200
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_CDSegNet_ScanNet200.py

# Testing on nuScenes
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/test_CDSegNet_nuScenes.py

# Testing for time
# Please ensure:
#  a. inference on an NVIDIA GPU
#  b. no test-time augmentation (TTA) (aug_transform=[])
#  c. no fragmented inference (grid_size=0.0001)
CUDA_VISIBLE_DEVICES=0 python tools/test_time.py
```

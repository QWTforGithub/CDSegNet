# CDSegNet

This repo is the official project repository of the paper **_An End-to-End Robust Point Cloud Semantic Segmentation Network with Single-Step Conditional Diffusion Models_**. 
 - [ [arXiv](https://arxiv.org/abs/2411.16308) ]
 - **_We plan to release the full code in the future, only releasing the relevant training logs._**
## The Overall Framework 
<img src="assets/CDSegNet.png" alt="cdsegnet" width="900"/> 

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
cd CDSegNet-main
sh compile.sh

# install flashattention
# 1. cuda11.8 -> cuda11.6
#   vim ~/.bashrc
#   export PATH="/usr/local/cuda-11.8/bin:$PATH" -> export PATH="/usr/local/cuda-11.6/bin:$PATH"
#   export CUDA_HOME="/usr/local/cuda-11.8" -> export CUDA_HOME="/usr/local/cuda-11.6"
#   source ~/.bashrc
# 2. Please download flash_attn-2.5.7+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
#   a. The official website: https://github.com/Dao-AILab/flash-attention/releases?page=2
#   b. Our links: Baidu Netdisk and Google Drive in Model Zoo
# 3. pip install flushattention flash_attn-2.5.7+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

## Data Preparation
...

## Model Zoo
<img src="assets/compare.png" alt="compare" width="900"/> 

### Indoor Benchmark
| Model | Benchmark | Only Training Data? | Num GPUs | Val mIoU | Test mIoU | checkpoint |
| :---: | :---: |:---------------:| :---: | :---: | :---: | :---: |
| CDSegNet | ScanNet |     &check;     | 1,2,4 | 77.9% | 74.5% | [Link1](https://pan.baidu.com/s/1SrcC710jDxSugvyfqzA_pQ?pwd=jxnr), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 + CNF | ScanNet |     &check;     | 2,4 | 77.7% | 73.9% |  [Link1](https://pan.baidu.com/s/1SrcC710jDxSugvyfqzA_pQ?pwd=jxnr), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 | ScanNet |     &check;     | 4 | 77.6% | 73.6% |  [Link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/scannet-semseg-pt-v3m1-0-base/model) |
| CDSegNet | ScanNet200 |     &check;     | 2,4 | 36.3% | 34.1% |  [Link1](https://pan.baidu.com/s/1SrcC710jDxSugvyfqzA_pQ?pwd=jxnr), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 + CNF | ScanNet200 | &check;  | 2,4 | 35.9% | 33.7% | [Link1](https://pan.baidu.com/s/1SrcC710jDxSugvyfqzA_pQ?pwd=jxnr), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 | ScanNet200 | &check;  | 4 | 35.3% | 33.2% | [Link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/scannet200-semseg-pt-v3m1-0-base/model) |

### Outdoor Benckmark
| Model | Benchmark | Only Training Data? | Num GPUs | Val mIoU | Test mIoU | checkpoint |
| :---: | :---: |:---------------:| :---: | :---: | :---: | :---: |
| CDSegNet | nuScenes |     &check;     | 4 | 81.2% | 82.0% | [Link1](https://pan.baidu.com/s/1SrcC710jDxSugvyfqzA_pQ?pwd=jxnr), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 + CNF | nuScenes |     &check;     | 4 | 81.0% | 82.8% |  [Link1](https://pan.baidu.com/s/1SrcC710jDxSugvyfqzA_pQ?pwd=jxnr), [Link2](https://drive.google.com/drive/folders/1iRS5hMci8ZWW4uGYTmmXCBA-wwjjbjTW?usp=sharing) |
| PTv3 | nuScenes |     &check;     | 4 | 80.3% | 81.2% |  [Link1](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/nuscenes-semseg-pt-v3m1-0-base/model) |


## Quick Start
### Example
...

### Training

### Testing
...

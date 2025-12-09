# MFA-Pose

*Enhancing Human Pose Estimation via Multi-Scale
Feature Fusion and Attention Mechanisms*

## Introduction

This repository contains the implementation for the work presented in *Enhancing Human Pose Estimation via Multi-Scale
Feature Fusion and Attention Mechanisms*.

The **training pipeline, model architectures, and evaluation protocols** of this project are **fully consistent with** [MMPose](https://github.com/open-mmlab/mmpose), the open-source toolbox for pose estimation based on PyTorch.

## Installation & Quick Start

Please follow the official [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html) to set up the environment. Our code is built upon the MMPose framework.

A typical installation command is:
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"  # Required for top-down models
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
```

## Data Download

The experiments in our paper are conducted on the following standard benchmarks. Please download them from their official websites:

COCO Dataset: http://cocodataset.org

MPII Human Pose Dataset: http://human-pose.mpi-inf.mpg.de

CrowdPose Dataset: https://github.com/Jeff-sjtu/CrowdPose

For data preparation details, please refer to the MMPose Dataset Preparation Documentation.

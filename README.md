# CNN Computer Vision Projects

A collection of deep learning projects focused on image classification using Convolutional Neural Networks (CNNs) built with PyTorch.

![Australian Road Sign Classification](https://i.imgur.com/JONwDM0.png)

## Projects

### 1. Australian Road Sign Classification
A CNN model achieving 93.48% accuracy in classifying Australian road signs into 5 categories:
- Keep Left
- No Left Turn
- No Right Turn
- Pedestrian Crossing
- Speed Limit

**Key Features:**
- Custom CNN architecture with progressive feature extraction (3→32→64→128 channels)
- Dataset of 4,399 images from Melbourne driving footage
- Strong per-class performance (F1-scores ranging from 88.89% to 96.15%)

### 2. CIFAR-10 Classification
A CNN implementation achieving 82.61% accuracy on the CIFAR-10 dataset, featuring:
- Comprehensive data augmentation pipeline
- Batch normalization and dropout regularization
- Progressive feature extraction architecture

## Requirements
- Python 3.x
- PyTorch >= 2.0.0
- torchvision
- matplotlib
- numpy
- seaborn
- scikit-learn

## Getting Started
Each project contains detailed documentation in its respective directory:
- [Road Sign Classification](./roadsign-classification-cnn)
- [CIFAR-10 Classification](./cifar10-cnn-pytorch)

## Results
| Project | Accuracy | F1-Score |
|---------|----------|-----------|
| Road Signs | 93.48% | 92.98% |
| CIFAR-10 | 82.61% | - |

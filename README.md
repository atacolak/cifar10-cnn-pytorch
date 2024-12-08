# CIFAR-10 Image Classification with CNN

A PyTorch implementation of a Convolutional Neural Network (CNN) for the CIFAR-10 image classification task. The model achieves 82.61% accuracy using a custom architecture with batch normalization and data augmentation.

![CNN Architecture](https://i.imgur.com/v68q26s.jpeg)

## Overview

This project implements a CNN to classify images from the CIFAR-10 dataset into 10 different categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Architecture

The CNN uses a progressive feature extraction pattern:
- Three convolutional blocks (32 -> 64 -> 128 filters)
- Batch normalization after each convolution
- Dropout regularization (25%)
- Three fully connected layers (512 -> 128 -> 10)

## Requirements

```
python
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
seaborn>=0.12.0
scikit-learn>=1.2.0
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/atacolak/cifar10-cnn-pytorch
cd cifar10-cnn-pytorch
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter cnn.ipynb
```

## Data Preprocessing

Training data augmentation includes:
- Random horizontal flips
- Random rotations (+/- 10°)
- Affine transformations
- Color jittering
- Normalization (mean=0.5, std=0.5)

## Training

The model is trained with:
- Adam optimizer (lr=0.001, weight_decay=1e-4)
- ReduceLROnPlateau scheduler
- Cross-entropy loss
- Batch size of 128

## Results

- Final validation accuracy: 82.61%
- Training duration: 50 epochs


## Acknowledgments

- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch documentation: https://pytorch.org/docs/stable/index.html

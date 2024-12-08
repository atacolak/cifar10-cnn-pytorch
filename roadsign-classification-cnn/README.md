# Road Sign Classification using CNN

A deep learning model for classifying Australian road signs into 5 different categories using a Convolutional Neural Network (CNN) built with PyTorch.

![roadsign-classification](https://i.imgur.com/JONwDM0.png)

## Overview

This project implements a custom CNN architecture to classify road signs into the following categories:
- Keep Left
- No Left Turn
- No Right Turn
- Pedestrian Crossing
- Speed Limit

## Dataset

The dataset consists of road sign images captured from Melbourne driving footage:
- Training: 3,826 images
- Validation: 388 images
- Testing: 185 images

### Data Distribution
- Keep Left: 618 images
- No Left Turn: 366 images
- No Right Turn: 774 images
- Pedestrian Crossing: 279 images
- Speed Limit: 1,788 images

### Preprocessing
- Images are resized to 224x224 pixels
- Custom normalization using dataset statistics:
  - Mean: [0.477, 0.444, 0.436]
  - Std: [0.137, 0.143, 0.144]

## Model Architecture

The CNN architecture consists of:
1. Three convolutional layers:
   - Conv1: 3→32 channels, 3x3 kernel
   - Conv2: 32→64 channels, 3x3 kernel
   - Conv3: 64→128 channels, 3x3 kernel
2. Max pooling after each convolutional layer
3. Two fully connected layers:
   - FC1: 128*28*28 → 256
   - FC2: 256 → 5 (output layer)
4. ReLU activation functions

## Performance

The model achieves strong performance across all classes:

### Overall Metrics
- Accuracy: 93.48%
- Precision: 94.56%
- Recall: 91.66%
- F1-Score: 92.98%

### Class-Specific Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|-----------|
| Keep Left | 100.00% | 92.59% | 96.15% |
| No Left Turn | 90.91% | 86.96% | 88.89% |
| No Right Turn | 88.89% | 94.12% | 91.43% |
| Pedestrian Crossing | 100.00% | 88.24% | 93.75% |
| Speed Limit | 93.02% | 96.39% | 94.67% |
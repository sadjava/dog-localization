# Dog Localization using Neural Network 
Localize dog on images in ImageNet dataset using custom built convolutional neural network.
Demo: https://huggingface.co/spaces/sadjava/dog-localization

## Features
⚡ Image Localization
⚡ Pretrained Model
⚡ Transfer Learning

## Limitations
❌ Cannot handle multiple objects

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Criteria](#evaluation-criteria)
- [Solution Approach](#solution-approach)

## Objective
I'll build a neural network using PyTorch. The goal here is to build a system to localize the dog on a given image if there is one. 

## Dataset
- Dataset consists of 20,000 images with dogs and 5,000 images without.
- For each image containing a dog, bounding box is presented. Bounding box is given as four numbers: x and y coordinates of the center, width and height of the box.
- Each image in the dataset is an RGB image of arbitrary size.


## Evaluation Criteria

### Loss Function  
Binary Cross Entropy Loss for predicting the presence of a dog and Smooth L1 Loss for predicting bounding box are used as the loss functions during model training and validation.

### Performance Metric
`IoU` is used as the model's performance metric on the test-set.


## Solution Approach
- Training dataset along with testing dataset are taken from ImageNet dataset, as images and annotations of classes of dog breeds.
- Training dataset is then further split into training (20,000 samples) and validation (5,000) samples sets
- The training, validation, and testing datasets are then wrapped in PyTorch `DataLoader` object so that we can iterate through them with ease. A `batch_size` can be configured.


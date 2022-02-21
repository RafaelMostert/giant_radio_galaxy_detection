#!/usr/bin/env python
# coding: utf-8
"""
Script to determine mean and std of our dataset
Output should manually be placed inside training and inference scripts.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
from collections import Counter
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
import socket
from tqdm import tqdm


# # Load Data
dataset_name = 'cutouts_res6arcsec_destsize350arcsec_nfields100'
model_name='resnet101'
data_inspection=False

start = time.time()
print('-'*80)
print('Giants Resnet training script')
print('-'*80)

hostname = socket.gethostname()
print("Data- and save-paths set based on host:", hostname)
if hostname.startswith('lgm4'):
    base_path = '/data1/mostertrij/data/giants' 
elif hostname.endswith('liacs.nl'):
    base_path = '/data/mostertrij/data/giants' 
elif hostname.startswith('kafka'):
    base_path = '/home/rafael/data/mostertrij/data/giants' 
else:
    print("Edit this script to include the correct paths for your machine:", hostname)
    quit()
data_dir = os.path.join(base_path, dataset_name)
trained_dir = os.path.join(base_path, 'trained_models')
os.makedirs(trained_dir,exist_ok=True)
print("Assuming dataset is located at:", data_dir)
print("Saving trained models at:", trained_dir)

# Data augmentation and normalization for training
# Just normalization for validation
print("\nLoad data")
image_dimension_before_rotation = 400
image_dimension = int(np.floor(image_dimension_before_rotation/np.sqrt(2)))
print("Image dimension before and after rotation in pixels:", image_dimension_before_rotation,
        image_dimension)
"""
torchvision.transforms.Normalize(mean, std, inplace=False)[source]
Normalize a tensor image with mean and standard deviation. 
This transform does not support PIL Image. 
Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels, 
this transform will normalize each channel of the input torch.
*Tensor i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]
"""

data_transforms = {
    'train': transforms.Compose([
        # Transforms for the giants
        transforms.Resize(image_dimension_before_rotation),
        transforms.CenterCrop(image_dimension),
        #transforms.RandomResizedCrop(image_dimension),
        #    #interpolation=<InterpolationMode.NEAREST: 'nearest'>, ),
        #transforms.RandomGrayscale(p=1), # For BEES only!
        transforms.ToTensor(),

    ]),
    'val': transforms.Compose([
        transforms.Resize(image_dimension_before_rotation),
        transforms.CenterCrop(image_dimension),
        #transforms.RandomGrayscale(p=1), # For BEES only!
        transforms.ToTensor(),
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'],
    batch_size=4,shuffle=True),
             'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,shuffle=True)}

nimages = 0
mean = 0.0
var = 0.0
sample_len = int(len(dataloaders['train'])/4)
for i_batch, batch_target in tqdm(enumerate(dataloaders['train']),
        total=int(sample_len)):
    batch = batch_target[0]
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0)
    var += batch.var(2).sum(0)
    if i_batch*4 > sample_len:
        break

mean /= nimages
var /= nimages
std = torch.sqrt(var)
print("Per channel mean and std.", mean, std)
print("To be used for normalizing the dataset during training and inference")

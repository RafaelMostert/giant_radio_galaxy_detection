#!/usr/bin/env python
# coding: utf-8
"""
Script to train a resnet
to determine if a Stokes-I radio cutout
contains a giant radio galaxy candidate.

Copyright (c) 2022 Rafael Mostert
See LICENSE.md in root directory for full BSD-3 license.

Adapted from
Author: Sasank Chilamkurthy
License: BSD
Source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
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
data_mean = [0.2460, 0.6437, 0.4650]
data_std = [0.1285, 0.1169, 0.0789]
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # Transforms for the giants
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation((-180,180),expand=False),
        transforms.Resize(image_dimension_before_rotation),
        transforms.CenterCrop(image_dimension),
        #transforms.RandomResizedCrop(image_dimension),
        #    #interpolation=<InterpolationMode.NEAREST: 'nearest'>, ),
        #transforms.RandomGrayscale(p=1), # For BEES only!
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(data_mean, data_std )

    ]),
    'val': transforms.Compose([
        transforms.Resize(image_dimension_before_rotation),
        transforms.CenterCrop(image_dimension),
        #transforms.RandomGrayscale(p=1), # For BEES only!
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(data_mean, data_std )
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# total_num_images = np.sum([len(image_datasets[x]) 
#                                              for x in ['train', 'val']])
# weights = {x: total_num_images/len(image_datasets[x])
#                   for x in ['train', 'val']}



target_list = [t for _, t in image_datasets['train'].samples]
target_dict = Counter(target_list)
print(target_dict)
class_weights = [1/target_dict[t] for t in target_list]
target_list = torch.tensor(target_list)

weighted_sampler = WeightedRandomSampler(
    weights=class_weights,
    num_samples=len(image_datasets['train']),
    replacement=True)

dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4,
                    sampler=weighted_sampler,num_workers=4),
             'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,
                    shuffle=True, num_workers=4)}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = data_std * inp + data_mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(15,5))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.savefig(0.001)  # pause a bit so that plots are updated


if data_inspection:
    print(f"Showing training input examples (data_inspection={data_inspection})")
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
else:
    print(f"Not showing training input examples (data_inspection={data_inspection})")


# # Training the model

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(trained_dir,
                    f'model_weights_{model_name}_{dataset_name}_{datetime.today().date().isoformat()}.pth'))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# In[8]:


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(8,8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if model_name == 'resnet101':
    print("\nCreating a resnet101 model and load pretrained weights")
    model_ft = models.resnet101(pretrained=True)
else:
    print("\nCreating a resnet18 model and load pretrained weights")
    model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)


# # Train

print("\nTrain model")
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)
print("\nSave final model")
torch.save(model_ft.state_dict(), os.path.join(trained_dir,
    f'final_model_weights_{model_name}_{dataset_name}_{datetime.now().isoformat()}.pth'))

print(f"Done. Time taken: {time.time()-start:.1f} sec.")

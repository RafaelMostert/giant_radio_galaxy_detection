#!/usr/bin/env python
# coding: utf-8
"""
Script to visualise the predictions of a Conv net
trained to determine if a Stokes-I radio cutout
contains a giant radio galaxy candidate.
Includes an attempt at attributing which features
of the input image are predominantly responsible for triggering
the class prediction.

Copyright (c) 2022 Rafael Mostert
See LICENSE.md in root directory for full BSD-3 license.

Adapted from
Author: Sasank Chilamkurthy
License: BSD
Source: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
and
Author: Pytorch team 2019, Captum
License: BSD-3
Source:
https://captum.ai/tutorials/Resnet_TorchVision_Interpret
https://captum.ai/tutorials/Resnet_TorchVision_Ablation
"""

# Imports
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from collections import Counter
import torch.nn.functional as F

from matplotlib.colors import LinearSegmentedColormap

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import socket



start = time.time()
print('-'*80)
print('Giants Resnet inference and prediction attribution')
print('-'*80)
print('\nLoad preprocessed data which is ready to enterthe pytorch dataloader.')
hostname = socket.gethostname()
print("Data- and save-paths set based on host:", hostname)

# User defined hyperparameters
n_predictions = 100
nt_samples_batch_size=2
n_steps=100
internal_batch_size=None
if hostname.startswith('lgm4'):
    base_path = '/data1/mostertrij/data/giants' 
elif hostname.endswith('liacs.nl'):
    base_path = '/data/mostertrij/data/giants' 
elif hostname.startswith('kafka'):
    base_path = '/home/rafael/data/mostertrij/data/giants' 
    nt_samples_batch_size=-1 # This device has low CUDA memory
    internal_batch_size=10
    n_predictions = 3
else:
    print("Edit this script to include the correct paths for your machine:", hostname)
    quit()

dataset_name = 'cutouts_res6arcsec_destsize350arcsec_nfields100'
data_dir = os.path.join(base_path, dataset_name)
print("Assuming dataset is located at:", data_dir)
attribution_dir = os.path.join(base_path,'attribution')
os.makedirs(attribution_dir,exist_ok=True)

print(f"About to generate visualisations for {n_predictions} cutouts in:", attribution_dir)

# Choose which trained model to use
resnet101 = True
if resnet101:
    trained_model_path = os.path.join(base_path,'trained_models',
            'model_weights_resnet101_cutouts_res6arcsec_destsize350arcsec_nfields100_2022-02-03.pth')
    save_prefix= 'resnet101_'
    # number of samples used in noise tunneling attribution
    nt_samples=20 
else:
    trained_model_path = os.path.join(base_path,'trained_models',
            'final_model_weights_cutouts_res6arcsec_destsize350arcsec_nfields100_2022-02-01.pth')
    save_prefix= 'resnet18_'
    # number of samples used in noise tunneling attribution
    nt_samples=20
print("Using trained model:", trained_model_path)

# Data augmentation and normalization for training
# Just normalization for validation
image_dimension_before_rotation = 300
image_dimension = int(np.floor(image_dimension_before_rotation/np.sqrt(2)))
#TODO: Use mean and std specific to our dataset
"""
torchvision.transforms.Normalize(mean, std, inplace=False)[source]
Normalize a tensor image with mean and standard deviation. 
This transform does not support PIL Image. 
Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels, 
this transform will normalize each channel of the input torch.
*Tensor i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]
"""

# Transforms for the giants
data_mean = [0.2460, 0.6437, 0.4650]
data_std = [0.1285, 0.1169, 0.0789]
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation((-180,180),expand=False),
        transforms.Resize(image_dimension_before_rotation),
        transforms.CenterCrop(image_dimension),
        #transforms.RandomResizedCrop(image_dimension),
        #    #interpolation=<InterpolationMode.NEAREST: 'nearest'>, ),
        #transforms.RandomGrayscale(p=1), # For BEES only!
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std )

    ]),
    'val': transforms.Compose([
        transforms.Resize(image_dimension_before_rotation),
        transforms.CenterCrop(image_dimension),
        #transforms.RandomGrayscale(p=1), # For BEES only!
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std )
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

targets_list = [t for _, t in image_datasets['train'].samples]
targets_dict = Counter(targets_list)
class_weights = [1/targets_dict[t] for t in targets_list]
targets_list = torch.tensor(targets_list)

# Address our class imbalance
# (We have way more cutouts without than with giants)
weighted_sampler = WeightedRandomSampler(
    weights=class_weights,
    num_samples=len(image_datasets['train']),
    replacement=True)

dataloaders = {'train': DataLoader(image_datasets['train'], batch_size=4,
                    sampler=weighted_sampler,num_workers=4),
             'val': DataLoader(image_datasets['val'], batch_size=4,
                    shuffle=True, num_workers=4)}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('\nCreate model architecture and load trained weights')
if resnet101:
    model = models.resnet101(pretrained=False)
else:
    model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(trained_model_path))
model = model.to(device)


print('Inference')

def input_to_vis(inp):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    inp = data_std * inp + data_mean
    inp = np.clip(inp, 0, 1)
    return inp

def imshow_vis(ax,inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = data_std * inp + data_mean
    inp = np.clip(inp, 0, 1)
    #plt.figure()#figsize=(15,5))
    ax.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.show()

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
            n_in_batch = inputs.size()[0]
            f,axes = plt.subplots(1,n_in_batch, figsize=(15,6))

            #print("outputs, preds, inputs", outputs.size(), preds.size()[0],inputs.size())
            for j,ax in enumerate(axes):
                images_so_far += 1
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow_vis(ax,inputs.cpu().data[j])
            plt.show()
            if images_so_far > 10:
                model.train(mode=was_training)
                plt.close('all')
                return
        #model.train(mode=was_training)
    plt.close('all')

print('\nModel attribution')
# Set model to eval mode
model = model.eval()

default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)
idx_to_class = dict([(value, key) for key, value in 
                         dataloaders['train'].dataset.class_to_idx.items()])

# get input
img_idx = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['val']):

        inputs = inputs.to(device)
        labels = labels.to('cpu')
        
        # Predict the class of the input image
        outputs = model(inputs)
        something, preds = torch.max(outputs, 1)
        n_in_batch = inputs.size()[0]

        softmax_outputs = F.softmax(outputs, dim=1)
        prediction_scores, pred_label_idxs = torch.topk(softmax_outputs, 1)
        pred_label_idxs.squeeze_()


        predicted_labels = [idx_to_class[pred_label_idx.item()] 
                            for pred_label_idx in pred_label_idxs]


        # for each image in the mini batch...
        for input_idx, (predicted_label,pred_label_idx, prediction_score, output, actual_label) in enumerate(zip(
                    predicted_labels, pred_label_idxs,prediction_scores,outputs,labels)):

            print(f'Predicted: {predicted_label} ({prediction_score.squeeze().item():.2f})')
            input = inputs[input_idx]
            input = input[None, :]

            # Calculate the integrated gradients for class prediction attribution
            transformed_img = input_to_vis(inputs[input_idx])
            integrated_gradients = IntegratedGradients(model)


            attributions_ig = integrated_gradients.attribute(input,
                    internal_batch_size=internal_batch_size,
                                                target=pred_label_idx, n_steps=n_steps)

            fig,ax = viz.visualize_image_attr_multiple(
                    np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                         transformed_img,
                                         ["original_image",'heat_map'],
                                         ['all','positive'],
                                         cmap=default_cmap,
                                         show_colorbar=True,
                                         outlier_perc=1)
            fig.suptitle(f'Attributions\nPredicted: {predicted_label} ({prediction_score.squeeze().item():.2f}) Actual label: {idx_to_class[actual_label.item()]}')
            plt.savefig(os.path.join(attribution_dir,f"{save_prefix}attributions_{img_idx:03}.png")) 
            plt.close()



            del attributions_ig
            torch.cuda.empty_cache()

            # Calculate the noise tunneled integrated gradients for class prediction attribution
            if nt_samples_batch_size>0:
                noise_tunnel = NoiseTunnel(integrated_gradients)

                attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=nt_samples,
                        nt_samples_batch_size=nt_samples_batch_size, nt_type='smoothgrad_sq',
                        target=pred_label_idx)
                fig,ax = viz.visualize_image_attr_multiple(
                        np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                        transformed_img,
                                                      ["original_image", "heat_map"],
                                                      ["all", "positive"],
                                                      cmap=default_cmap,
                                                      show_colorbar=True)

                fig.suptitle(f'Noise tunnelled attributions\nPredicted: {predicted_label} ({prediction_score.squeeze().item():.2f}) Actual label: {idx_to_class[actual_label.item()]}')
                plt.savefig(os.path.join(attribution_dir,
                    f"{save_prefix}attributions_{img_idx:03}_noisetunnel.png"))
                plt.close('all')
            img_idx+=1
            if img_idx>n_predictions:
                break
        if img_idx>n_predictions:
            break

print(f"Done. Time taken: {time.time()-start:.1f} sec.")

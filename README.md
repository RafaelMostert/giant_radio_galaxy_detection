# Giant radio galaxy detection
We use pytorch to train a resnet
to determine if a Stokes-I radio cutout
contains a giant radio galaxy candidate.

We use captum to estimate which 
parts/pixels of the input contribute most to
the model's prediction.

Contents
* notebooks (unlisted): Contains Jupyter notebooks to interactively inspect our preprocessed dataset.
* scripts: Contains python scripts used to train a model and perform inference.
* training_annotations (unlisted): Contains giant radio source catalogues from which we derived the labels for our dataset.

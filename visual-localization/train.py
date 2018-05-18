#!/usr/bin/env python3

import torch

from dataset import DeepLocAugmented, make_train_valid_loader, make_test_loader
from network import GoogLeNet

# Load the dataset
train_data = DeepLocAugmented("train")
print("Training set size: {} samples".format(len(train_data)))

# Generate the data loaders
train_loader, valid_loader = make_train_valid_loader(train_data, valid_percentage = 0.2)

# Define the model
net = GoogLeNet()
net.cuda()

# Training phase
for images, xs, qs in train_loader:
    ps = torch.cat([xs, qs], dim = 1)

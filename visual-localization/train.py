#!/usr/bin/env python3

import torch

from dataset import DeepLocAugmented, make_train_valid_loader, make_test_loader
from network import parameters, PoseNet

# Load the dataset
train_data = DeepLocAugmented("train")
print("Training set size: {} samples".format(len(train_data)))

# Generate the data loaders
train_loader, valid_loader = make_train_valid_loader(train_data, valid_percentage = 0.2)

# Define the model
net = PoseNet()
net.cuda()

# Check the number of parameters
trainable_params, total_params = parameters(net)
print("Trainable parameters: {}".format(trainable_params))
print("Total parameters: {}".format(total_params))
print("Memory requirement: {} MiB".format(((total_params * 8) / 1024) / 1024))

# Training phase
for images, xs, qs in train_loader:
    ps = torch.cat([xs, qs], dim = 1).cuda()
    images = images.cuda()

    # TODO Predict the pose
    ps_out1, ps_out2, ps_out3 = net(images)
    print(ps)
    print(ps_out1)
    print(ps_out2)
    print(ps_out3)
    input()

    # TODO Do a backpropagation step

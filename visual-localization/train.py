#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import DeepLocAugmented, make_train_valid_loader, make_test_loader
from network import parameters, PoseNet
from customized_loss import Customized_Loss

# Device - use CPU is CUDA is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the dataset
train_data = DeepLocAugmented("train")
print("Training set size: {} samples".format(len(train_data)))

# Generate the data loaders
train_loader, valid_loader = make_train_valid_loader(train_data, valid_percentage = 0.2, batch_size = 16)

# Define the model
net = PoseNet()
net.to(device = device)

# Check the number of parameters
trainable_params, total_params = parameters(net)
print("Trainable parameters: {}".format(trainable_params))
print("Total parameters: {}".format(total_params))
print("Memory requirement: {} MiB".format(((total_params * 8) / 1024) / 1024))

# TODO Optimizer
optimizer = optim.SGD(net.parameters(), lr = 1e-5, momentum = 0.9)

# TODO Loss function
criterion = Customized_Loss()

# TODO Training phase
for epoch in range(100):

    net.train()
    for images, xs, qs in train_loader:
        ps = torch.cat([xs, qs], dim = 1).to(device = device)
        images = images.to(device = device)

        # TODO Predict the pose
        ps_out1, ps_out2, ps_out3 = net(images)
        loss1 = criterion(ps_out1, ps)
        loss2 = criterion(ps_out2, ps)
        loss3 = criterion(ps_out3, ps)
        print("Loss: {}".format(loss3))

        # TODO Do a backpropagation step
        optimizer.zero_grad()
        loss1.backward(retain_graph = True)
        loss2.backward(retain_graph = True)
        loss3.backward()
        optimizer.step()

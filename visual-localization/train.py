#!/usr/bin/env python3

import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from dataset import DeepLocAugmented, make_train_valid_loader, make_test_loader
from network import parameters, PoseNet
from customized_loss import Customized_Loss
from utils import print_torch_cuda_mem_usage, Stopwatch

# Device - use CPU is CUDA is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the dataset
train_data = DeepLocAugmented("train")
print("Training set size: {} samples".format(len(train_data)))

# Generate the data loaders
train_loader, valid_loader = make_train_valid_loader(train_data, valid_percentage = 0.2)

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

# Loss function
BETA = 250
criterion = Customized_Loss(beta = BETA)

# Plot variables
x = []
y_training = []
y_loss = []

# Time measuring
stopwatch = Stopwatch()

# TODO Training phase
for epoch in range(100):
    x.append(epoch + 1)
    print("[Epoch {}]".format(epoch + 1))

    net.train()
    total_loss = 0
    num_iters = 0
    stopwatch.start()
    for images, ps in train_loader:
        ps = ps.to(device = device)
        images = images.to(device = device)

        # Predict the pose
        ps_out1, ps_out2, ps_out3 = net(images)
        loss1 = criterion(ps_out1, ps)
        loss2 = criterion(ps_out2, ps)
        loss3 = criterion(ps_out3, ps)

        total_loss += loss3.item() # Important to use .item() !
        num_iters += 1

        # Do a backpropagation step
        optimizer.zero_grad()
        loss1.backward(retain_graph = True)
        loss2.backward(retain_graph = True)
        loss3.backward()
        optimizer.step()

        if num_iters % 30 == 0:
            print("{:6.2f} %".format(num_iters * 100 / len(train_loader)), end = "\r")

    # Print some stats
    elapsed_time = stopwatch.stop()
    print("Epoch time: {:0.2f} minutes".format(elapsed_time))
    print_torch_cuda_mem_usage()

    # Save the average epoch loss
    avg_loss = total_loss / num_iters
    print("Average loss: {}".format(avg_loss))
    y_loss.append(avg_loss)

# Plot the average loss over the epochs
loss_fig = plt.figure()
loss_ax = loss_fig.gca()
loss_ax.plot(x, y_loss, "b")
plt.xlabel("Epoch")
plt.ylabel("Average loss")
loss_ax.grid(True)

# Save the diagrams and the model
timestamp = datetime.datetime.now()
identifier = "model_{}_{}_{}_{}_{}_{}".format(BETA, timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)

loss_fig.savefig("{}/{}.loss.png".format(directory, identifier))

model_path = "{}/{}".format(directory, identifier)
torch.save(net.state_dict(), model_path)
print("Model parameters saved to {}.".format(model_path))

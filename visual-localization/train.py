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

# TODO Loss function
criterion = Customized_Loss()

# Plot variables
x = []
y_training = []
y_loss = []

# TODO Training phase
for epoch in range(100):
    x.append(epoch + 1)
    print("[Epoch {}]".format(epoch + 1))

    net.train()
    total_loss = 0
    num_iters = 0
    for images, xs, qs in train_loader:
        ps = torch.cat([xs, qs], dim = 1).to(device = device)
        images = images.to(device = device)

        # TODO Predict the pose
        ps_out1, ps_out2, ps_out3 = net(images)
        loss1 = criterion(ps_out1, ps)
        loss2 = criterion(ps_out2, ps)
        loss3 = criterion(ps_out3, ps)

        total_loss += loss3
        num_iters += 1

        # TODO Do a backpropagation step
        optimizer.zero_grad()
        loss1.backward(retain_graph = True)
        loss2.backward(retain_graph = True)
        loss3.backward()
        optimizer.step()

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
loss_fig.savefig("avg_loss.png")

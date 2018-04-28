#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

from dataset import GTSRBTraining, GTSRBTest, train_valid_loader, test_loader
from preprocessing import Resize, SubtractMean, MakeTensor
from network import NeuralNetwork

NUM_EPOCHS = 10

# Transformations
shared_transform = transforms.Compose([
    Resize(32, 32),
    SubtractMean(),
    MakeTensor()
])

# Dataset
training_data = GTSRBTraining(
    training_path = "training",
    transform = shared_transform
)

test_data = GTSRBTest(
    test_path = "test",
    transform = shared_transform
)

print("Training samples:", len(training_data))
print("Test samples:", len(test_data))

# Loaders
train_loader, valid_loader = train_valid_loader(training_data, 0.2)
test_loader = test_loader(test_data)

# Model
net = NeuralNetwork()
net.cuda()

# Loss function and optimization method
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 3e-4)

# Plot variables
x_axis = []
y_axis = []

# Epochs
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    num_iters = 0
    print("[Epoch {}]".format(epoch))

    # Train
    net.train()
    for batch in train_loader:
        Xs, ys = batch
        Xs, ys = Variable(Xs.cuda()), Variable(ys.cuda())

        # Forward pass
        ys_hat = net(Xs)
        loss = cost(ys_hat, ys)
        total_loss += loss
        num_iters += 1

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimize
        optimizer.step()

    # Compute the average loss in the previous epoch
    print("Average loss: {}".format(total_loss / num_iters))

    # Validate
    net.eval()
    total_predictions = 0
    correct_predictions = 0
    for batch in valid_loader:
        Xs, ys = batch
        Xs, ys = Variable(Xs.cuda()), Variable(ys.cuda())
        ys_hat = net(Xs)
        _, predicted = torch.max(ys_hat.data, 1)
        total_predictions += ys.size(0)
        correct_predictions += (predicted == ys.data).sum().item()
    accuracy = correct_predictions * 100.0 / total_predictions
    print("Validation accuracy: {}%".format(accuracy))
    plt.plot(epoch, accuracy)
    x_axis.append(epoch)
    y_axis.append(accuracy)

# Plot validation accuracy over epochs
plt.plot(x_axis, y_axis)
plt.savefig("learning_curve.png")

# Evaluate on the test set
print("[TEST]")
net.eval()
total_predictions = 0
correct_predictions = 0
for batch in test_loader:
    Xs, ys = batch
    Xs, ys = Xs.cuda(), ys.cuda()
    ys_hat = net(Xs)
    _, predicted = torch.max(ys_hat.data, 1)
    total_predictions += ys.size(0)
    correct_predictions += (predicted == ys.data).sum().item()
accuracy = correct_predictions * 100.0 / total_predictions
print("Test accuracy: {}%".format(accuracy))

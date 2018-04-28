#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from dataset import GTSRBTraining, GTSRBTest, train_valid_loader, test_loader
from preprocessing import Resize, SubtractMean, MakeTensor
from network import NeuralNetwork

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

cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 3e-4)

for epoch in range(10):
    total_loss = 0
    num_iters = 0
    print("[Epoch {}]".format(epoch))

    # Train
    total_predictions = 0
    correct_predictions = 0
    net.train()
    for batch in train_loader:
        Xs, ys = batch
        Xs, ys = Variable(Xs.cuda()), Variable(ys.cuda())

        # Forward pass
        ys_hat = net(Xs)
        loss = cost(ys_hat, ys)
        total_loss += loss
        num_iters += 1

        _, predicted = torch.max(ys_hat.data, 1)
        total_predictions += ys.size(0)
        correct_predictions += (predicted == ys.data).sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimize
        optimizer.step()

    # Compute the average loss in the previous epoch
    print("Average loss: {}".format(total_loss / num_iters))
    print("Train accuracy:", correct_predictions * 100 / total_predictions)

    # Validate
    total_predictions = 0
    correct_predictions = 0
    net.eval()
    for batch in valid_loader:
        Xs, ys = batch
        Xs, ys = Variable(Xs.cuda()), Variable(ys.cuda())
        ys_hat = net(Xs)
        _, predicted = torch.max(ys_hat.data, 1)
        total_predictions += ys.size(0)
        correct_predictions += (predicted == ys.data).sum()
    print("Validation accuracy:", correct_predictions * 100 / total_predictions)

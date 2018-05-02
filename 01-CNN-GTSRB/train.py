#!/usr/bin/python3

import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from dataset import GTSRBTraining, make_train_valid_loader, evaluate
from network import NeuralNetwork
from transform import shared_transform

# Parse CLI arguments
args_parser = argparse.ArgumentParser(
	formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

args_parser.add_argument(
	"--num_epochs",
	type = int,
	help = "Number of training epochs",
    default = 15
)

args_parser.add_argument(
	"--eval_train",
	type = bool,
	help = "Evaluate the model on the training set after each epoch",
    default = False
)

args = args_parser.parse_args()

print("Training epochs: {}".format(args.num_epochs))

# Dataset
training_data = GTSRBTraining(
    training_path = "training",
    transform = shared_transform
)

print("Training samples:", len(training_data))

# Loaders
train_loader, valid_loader = make_train_valid_loader(training_data, valid_percentage = 0.2)

# Model
net = NeuralNetwork()
net.cuda()

# Loss function and optimization method
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 1e-4, momentum = 0.5)

# Plot variables
plt.hold(True)
x = []
y_validation = []
y_training = []

# Epochs
for epoch in range(args.num_epochs):
    x.append(epoch + 1)
    print("[Epoch {}]".format(epoch + 1))

    # Train
    net.train()
    for batch_idx, (Xs, ys) in enumerate(train_loader):
        Xs, ys = Xs.cuda(), ys.cuda()

        # Forward pass
        ys_hat = net(Xs)
        loss = cost(ys_hat, ys)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimize
        optimizer.step()

        # Report loss every 50 batches
        if batch_idx % 50 == 0:
            print("{} / {} => loss = {}".format(batch_idx * len(Xs), len(train_loader.dataset), loss.item()))

    if args.eval_train:
        # Evaluate on the training set
        accuracy = evaluate(net, train_loader)
        y_training.append(accuracy)
        print("Training accuracy: {}%".format(accuracy))

    # Evaluate on the validation set
    accuracy = evaluate(net, valid_loader)
    y_validation.append(accuracy)
    print("Validation accuracy: {}%".format(accuracy))

# Plot validation accuracy over epochs
plt.plot(x, y_training, "b", label = "Training")
plt.plot(x, y_validation, "r", label = "Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("learning_curve.png")

# Save the model
directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)

timestamp = datetime.datetime.now()

model_path = "{}/model_{}_{}_{}_{}_{}".format(directory, timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)
torch.save(net.state_dict(), model_path)
print("Model parameters saved to {}.".format(model_path))
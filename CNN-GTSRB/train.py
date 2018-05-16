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
from preprocessing import shared_transform

# Parse CLI arguments
args_parser = argparse.ArgumentParser(
	formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

args_parser.add_argument(
	"--learning_rate",
	type = float,
	help = "SGD learning rate",
    default = 1e-4
)

args_parser.add_argument(
	"--momentum",
	type = float,
	help = "SGD momentum",
    default = 0.5
)

args_parser.add_argument(
	"--num_epochs",
	type = int,
	help = "Number of training epochs",
    default = 20
)

args_parser.add_argument(
	"--eval_train",
	type = bool,
	help = "Evaluate the model on the training set after each epoch",
    default = False
)

args = args_parser.parse_args()

print("Learning rate: {}".format(args.learning_rate))
print("Momentum: {}".format(args.momentum))
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
optimizer = optim.SGD(net.parameters(), lr = args.learning_rate, momentum = args.momentum)

# Plot variables
x = []
y_validation = []
y_training = []
y_loss = []

# Epochs
for epoch in range(args.num_epochs):
    x.append(epoch + 1)
    print("[Epoch {}]".format(epoch + 1))

    # Train
    net.train()
    total_loss = 0
    num_iters = 0
    for batch_idx, (Xs, ys) in enumerate(train_loader):
        Xs, ys = Xs.cuda(), ys.cuda()

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

        # Report loss every 50 batches
        if batch_idx % 50 == 0:
            print("{} / {} => loss = {}".format(batch_idx * len(Xs), len(train_loader.dataset), loss.item()))

    if args.eval_train:
        # Evaluate on the training set
        accuracy = evaluate(net, train_loader)
        y_training.append(accuracy)
        print("Training accuracy: {}%".format(accuracy))

    # Save the average epoch loss
    y_loss.append(total_loss / num_iters)

    # Evaluate on the validation set
    accuracy = evaluate(net, valid_loader)
    y_validation.append(accuracy)
    print("Validation accuracy: {}%".format(accuracy))

# Plot validation accuracy over the epochs
acc_fig = plt.figure()
acc_ax = acc_fig.gca()
acc_fig.hold(True)
if args.eval_train:
    acc_ax.plot(x, y_training, "b", label = "Training")
acc_ax.plot(x, y_validation, "r", label = "Validation")
# plt.xticks(x)
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
acc_ax.grid(True)
acc_fig.legend()

# Plot the average loss over the epochs
loss_fig = plt.figure()
loss_ax = loss_fig.gca()
loss_ax.plot(x, y_loss, "b")
plt.xlabel("Epoch")
plt.ylabel("Average loss")
loss_ax.grid(True)

# Save the diagrams and the model
timestamp = datetime.datetime.now()
identifier = "model_{}_{}_{}_{}_{}".format(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)

acc_fig.savefig("{}/{}.lc.png".format(directory, identifier))
loss_fig.savefig("{}/{}.loss.png".format(directory, identifier))

model_path = "{}/{}".format(directory, identifier)
torch.save(net.state_dict(), model_path)
print("Model parameters saved to {}.".format(model_path))

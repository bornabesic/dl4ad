#!/usr/bin/env python3

import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from dataset import DeepLocAugmented, make_loader, evaluate
from network import parameters, PoseNetSimple
from customized_loss import Customized_Loss
from utils import print_torch_cuda_mem_usage, Stopwatch

# Parse CLI arguments
args_parser = argparse.ArgumentParser(
	formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

args_parser.add_argument(
	"--learning_rate",
	type = float,
	help = "SGD learning rate",
    default = 1e-5
)

args_parser.add_argument(
	"--momentum",
	type = float,
	help = "SGD momentum",
    default = 0.9
)

args_parser.add_argument(
	"--loss_beta",
	type = float,
	help = "Beta parameter used with the loss function",
    default = 250
)

args_parser.add_argument(
	"--batch_size",
	type = int,
	help = "Batch size",
    default = 16
)

args_parser.add_argument(
	"--epochs",
	type = int,
	help = "Number of training epochs",
    default = 50
)

args_parser.add_argument(
	"--use_model",
	type = str,
	help = "Path to the model to continue training on"
)

args = args_parser.parse_args()

# Parameters

print(args)

LEARNING_RATE = args.learning_rate
MOMENTUM = args.momentum
LOSS_BETA = args.loss_beta
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
MODEL_PATH = args.use_model

# Device - use CPU is CUDA is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the dataset
train_data = DeepLocAugmented("train")
valid_data = DeepLocAugmented("validation")
print("Train set size: {} samples".format(len(train_data)))
print("Validation set size: {} samples".format(len(valid_data)))

# Generate the data loaders
train_loader = make_loader(train_data, batch_size = BATCH_SIZE)
valid_loader = make_loader(valid_data)

# Define the model
net = PoseNetSimple()
if MODEL_PATH is not None:
    print("Using {}.".format(MODEL_PATH))
    net.load_state_dict(torch.load(MODEL_PATH))
net.to(device = device)

# Check the number of parameters
trainable_params, total_params = parameters(net)
print("Trainable parameters: {}".format(trainable_params))
print("Total parameters: {}".format(total_params))
print("Memory requirement: {:.2f} MiB".format(((total_params * 4) / 1024) / 1024))

# TODO Optimizer
optimizer = optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

# Loss function
criterion = Customized_Loss(beta = LOSS_BETA)

# Plot variables
x = []
y_training = []
y_loss_valid = []

# Time measuring
stopwatch_train = Stopwatch()
stopwatch_epoch = Stopwatch()

# Prepare environment for the snapshot
timestamp = datetime.datetime.now()
identifier = "model_{}_{}_{}_{}_{}_{}".format(LOSS_BETA, timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)

model_path = "{}/{}".format(directory, identifier)

# TODO Training phase
total_epoch_time = 0
stopwatch_train.start()
for epoch in range(EPOCHS):
    x.append(epoch + 1)
    print("[Epoch {}]".format(epoch + 1))

    net.train()
    total_loss = 0
    num_iters = 0
    stopwatch_epoch.start()
    for images, ps in train_loader:
        ps = ps.to(device = device)
        images = images.to(device = device)

        # Predict the pose
        ps_out = net(images)
        loss = criterion(ps_out, ps)

        total_loss += loss.item() # Important to use .item() !
        num_iters += 1

        # Do a backpropagation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if num_iters % 30 == 0:
            print("{:6.2f} %".format(num_iters * 100 / len(train_loader)), end = "\r")

    # Print some stats
    elapsed_time = stopwatch_epoch.stop()
    print("Epoch time: {:0.2f} minutes".format(elapsed_time))
    print_torch_cuda_mem_usage()
    total_epoch_time += elapsed_time
    avg_epoch_time = total_epoch_time / (epoch + 1)
    training_time_left = (EPOCHS - epoch - 1) * avg_epoch_time
    print("Training time left: ~ {:.2f} minutes ({:.2f} hours)".format(training_time_left, training_time_left / 60))

    # Save the average epoch loss
    avg_loss = total_loss / num_iters
    print("Average training loss: {}".format(avg_loss))

    # Evaluate on the validation set
    avg_loss_valid = evaluate(net, criterion, valid_loader, device)
    y_loss_valid.append(avg_loss_valid)
    print("Average validation loss: {}".format(avg_loss_valid))

    # Plot the average loss over the epochs
    loss_fig = plt.figure()
    loss_ax = loss_fig.gca()
    loss_ax.plot(x, y_loss_valid, "r", label = "Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    loss_ax.grid(True)
    loss_fig.legend()

    loss_fig.savefig("{}/{}.loss.png".format(directory, identifier))

    # Save the model
    torch.save(net.state_dict(), model_path)
    print("Model parameters saved to {}.".format(model_path))

# Elapsed training time
print("-----------------")
elapsed_time = stopwatch_train.stop()
print("Training time: {:0.2f} minutes".format(elapsed_time))

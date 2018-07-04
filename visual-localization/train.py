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

from dataset import DeepLocAugmented, make_loader, evaluate_median, PerceptionCarDataset, PerceptionCarDatasetMerged
import network
from network import parameters, PoseNetSimple
from customized_loss import Customized_Loss
from utils import print_torch_cuda_mem_usage, Stopwatch, Logger

# Parse CLI arguments
args_parser = argparse.ArgumentParser(
	formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

args_parser.add_argument(
	"--learning_rate",
	type = float,
	help = "Optimizer learning rate",
    default = 1e-5
)

args_parser.add_argument(
	"--gamma",
	type = float,
	help = "SGD learning rate decay factor",
    default = 0.9
)

args_parser.add_argument(
	"--decay_lr_every",
	type = float,
	help = "Number of epochs after which learning rate decay factor will be applied",
    default = 10
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
	"--architecture",
	type = str,
	help = "Neural network architecture to use",
    default = "PoseNetSimple"
)

args_parser.add_argument(
	"--use_model",
	type = str,
	help = "Path to the model to continue training on"
)

args = args_parser.parse_args()

# Parameters

LEARNING_RATE = args.learning_rate
GAMMA = args.gamma
LR_DECAY_EPOCHS = args.decay_lr_every
LOSS_BETA = args.loss_beta
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
ARCHITECTURE = args.architecture
MODEL_PATH = args.use_model
arch_class = getattr(network, ARCHITECTURE)

# Device - use CPU is CUDA is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Prepare the environment
timestamp = datetime.datetime.now()
identifier = "{}_{}_{}_{}_{}_{}_{}".format(ARCHITECTURE, LOSS_BETA, timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)

model_path = "{}/{}".format(directory, identifier)
logger = Logger("{}/{}.log.txt".format(directory, identifier), print_to_stdout = True)

# Load the dataset
train_data = PerceptionCarDatasetMerged(
    "PerceptionCarDataset",
    "PerceptionCarDataset2",
    mode = "train"
)
valid_data = PerceptionCarDatasetMerged(
    "PerceptionCarDataset",
    "PerceptionCarDataset2",
    mode = "validation",
    preprocess = PerceptionCarDataset.valid_preprocessing,
    augment = False
)

logger.log("Train set size: {} samples".format(len(train_data)))
logger.log("Validation set size: {} samples".format(len(valid_data)))

logger.log("Learning rate: {}".format(LEARNING_RATE))
logger.log("Beta: {}".format(LOSS_BETA))
logger.log("Batch size: {}".format(BATCH_SIZE))
logger.log("Epochs: {}".format(EPOCHS))
logger.log("Architecture: {}".format(ARCHITECTURE))
logger.log("Device: {}".format(device))

# Generate the data loaders
train_loader = make_loader(train_data, batch_size = BATCH_SIZE, num_workers = 4)
valid_loader = make_loader(valid_data, batch_size = 1, num_workers = 4)

# Define the model
net = arch_class()
if MODEL_PATH is not None:
    logger.log("Using {}.".format(MODEL_PATH))
    net.load_state_dict(torch.load(MODEL_PATH))
net.to(device = device)

# Check the number of parameters
trainable_params, total_params = parameters(net)
logger.log("Trainable parameters: {}".format(trainable_params))
logger.log("Total parameters: {}".format(total_params))
logger.log("Memory requirement: {:.2f} MiB".format(((total_params * 4) / 1024) / 1024))

# Optimizer
optimizer = optim.Adam(net.parameters(), lr = LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = LR_DECAY_EPOCHS, gamma = GAMMA)

# Loss function
criterion = Customized_Loss(beta = LOSS_BETA)

# Plot variables
x = []
y_training = []
y_loss_valid = []

# Time measuring
stopwatch_train = Stopwatch()
stopwatch_epoch = Stopwatch()

# Training phase
total_epoch_time = 0
stopwatch_train.start()
for epoch in range(EPOCHS):
    x.append(epoch + 1)
    logger.log("[Epoch {}]".format(epoch + 1))

    net.train()
    scheduler.step()
    total_loss = 0
    num_iters = 0
    stopwatch_epoch.start()
    for images, ps in train_loader:
        ps = ps.to(device = device)
        images = images.to(device = device)

        # Predict the pose
        ps_outs = net(images)

        losses = tuple(map(lambda ps_out: criterion(ps_out, ps), ps_outs))

        total_loss += losses[-1].item() # Important to use .item() !
        last_training_loss =  losses[-1].item()
        num_iters += 1
        print("Loss of last training batch: {}".format(last_training_loss))

        # Do a backpropagation step
        optimizer.zero_grad()
        for loss in losses[:-1]:
            loss.backward(retain_graph = True)
        losses[-1].backward()
        optimizer.step()

        if num_iters % 30 == 0:
            logger.log("{:6.2f} %".format(num_iters * 100 / len(train_loader)), end = "\r")

    # Print some stats
    elapsed_time = stopwatch_epoch.stop()
    logger.log("Epoch time: {:0.2f} minutes".format(elapsed_time))
    print_torch_cuda_mem_usage()
    total_epoch_time += elapsed_time
    avg_epoch_time = total_epoch_time / (epoch + 1)
    training_time_left = (EPOCHS - epoch - 1) * avg_epoch_time
    logger.log("Training time left: ~ {:.2f} minutes ({:.2f} hours)".format(training_time_left, training_time_left / 60))

    # Save the average epoch loss
    avg_loss = total_loss / num_iters
    logger.log("Loss of last training batch: {}".format(last_training_loss))
    logger.log("Average training loss: {}".format(avg_loss))

    # Evaluate on the validation set
    # avg_loss_valid = evaluate(net, criterion, valid_loader, device)
    # y_loss_valid.append(avg_loss_valid)
    # logger.log("Average validation loss: {}".format(avg_loss_valid))
    x_error_median, q_error_median, loss_median = evaluate_median(net, criterion, valid_loader, device)
    y_loss_valid.append(loss_median)
    y_training.append(last_training_loss)
    logger.log("Median validation error: {:.2f} m, {:.2f} °".format(x_error_median, q_error_median))
    logger.log("Median validation loss: {}".format(loss_median))

    # Plot the average loss over the epochs
    loss_fig = plt.figure()
    loss_ax = loss_fig.gca()
    loss_ax.plot(x, y_loss_valid, "r", label = "Validation")
    loss_ax.plot(x, y_training, "b", label = "Training")
    plt.xlabel("Epoch")
    plt.ylabel("Median loss")
    loss_ax.grid(True)
    loss_fig.legend()

    loss_fig.savefig("{}/{}.loss.png".format(directory, identifier))
    plt.close(loss_fig)

    # Save the model
    torch.save(net.state_dict(), model_path)
    logger.log("Model parameters saved to {}.".format(model_path))

# Elapsed training time
logger.log("-----------------")
elapsed_time = stopwatch_train.stop()
logger.log("Training time: {:0.2f} minutes".format(elapsed_time))
logger.close()

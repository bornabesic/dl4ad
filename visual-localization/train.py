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
import numpy as np
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
	"--only_front_camera",
	type = bool,
	help = "Use only front camera of the car as the input of the neural network",
    default = False
)

args_parser.add_argument(
	"--split",
	type = str,
	help = "What type of train / validation / test split files will be used",
    default = "manual"
)

args_parser.add_argument(
	"--learning_rate",
	type = float,
	help = "Optimizer learning rate",
    default = 1e-5
)

args_parser.add_argument(
	"--momentum",
	type = float,
	help = "SGD momentum",
    default = 0.9
)

args_parser.add_argument(
	"--gamma",
	type = float,
	help = "SGD learning rate decay factor",
    default = 1
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
    default = 1
)

args_parser.add_argument(
	"--batch_size",
	type = int,
	help = "Batch size",
    default = 32
)

args_parser.add_argument(
	"--epochs",
	type = int,
	help = "Number of training epochs",
    default = 300
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

SPLIT = args.split
ONLY_FRONT_CAMERA = args.only_front_camera
LEARNING_RATE = args.learning_rate
MOMENTUM = args.momentum
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

# Load the dataset
train_data = PerceptionCarDatasetMerged(
    "PerceptionCarDataset",
    "PerceptionCarDataset2",
    mode = "train",
    only_front_camera = ONLY_FRONT_CAMERA
)
valid_data = PerceptionCarDatasetMerged(
    "PerceptionCarDataset",
    "PerceptionCarDataset2",
    mode = "validation",
    preprocess = PerceptionCarDataset.valid_preprocessing,
    augment = False,
    only_front_camera = ONLY_FRONT_CAMERA,
    split = SPLIT
)

# Generate the data loaders
train_loader = make_loader(train_data, batch_size = BATCH_SIZE, num_workers = 4)
valid_loader = make_loader(valid_data, batch_size = 1, num_workers = 4)

# Define the model
if MODEL_PATH is not None:
    net = network.NeuralNetworkModel.load(MODEL_PATH, device)
    net.log("Using {}.".format(MODEL_PATH))
else:
    arch_class = network.get_model_class(ARCHITECTURE)
    net = arch_class(only_front_camera = ONLY_FRONT_CAMERA, split = SPLIT)

net.to(device = device)

# Summary
net.log("Train set size: {} samples".format(len(train_data)))
net.log("Validation set size: {} samples".format(len(valid_data)))

net.log("Learning rate: {}".format(LEARNING_RATE))
net.log("Momentum: {}".format(MOMENTUM))
net.log("Beta: {}".format(LOSS_BETA))
net.log("Batch size: {}".format(BATCH_SIZE))
net.log("Epochs: {}".format(EPOCHS))
net.log("Architecture: {}".format(ARCHITECTURE))
net.log("Device: {}".format(device))

# Check the number of parameters
trainable_params, total_params = parameters(net)
net.log("Trainable parameters: {}".format(trainable_params))
net.log("Total parameters: {}".format(total_params))
net.log("Memory requirement: {:.2f} MiB".format(((total_params * 4) / 1024) / 1024))

# Optimizer
optimizer = optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = LR_DECAY_EPOCHS, gamma = GAMMA)

# Loss function
criterion = Customized_Loss(beta = LOSS_BETA)

# Time measuring
stopwatch_train = Stopwatch()
stopwatch_epoch = Stopwatch()

# Training phase
total_epoch_time = 0
stopwatch_train.start()
for epoch in range(EPOCHS):
    net.log("[Epoch {}]".format(epoch + 1))

    net.train()
    scheduler.step()
    training_losses = []
    num_iters = 0
    stopwatch_epoch.start()
    import itertools
    #for images, ps in itertools.islice(train_loader, 10):
    for images, ps in train_loader:
        ps = ps.to(device = device)
        images = images.to(device = device)

        # Predict the pose
        ps_outs = net(images)
        losses = tuple(map(lambda ps_out: criterion(ps_out, ps), ps_outs))

        loss = losses[-1].item()
        training_losses.append(loss) # Important to use .item() !
        num_iters += 1

        # Do a backpropagation step
        optimizer.zero_grad()
        for loss in losses[:-1]:
            loss.backward(retain_graph = True)
        losses[-1].backward()
        optimizer.step()

        if num_iters % 30 == 0:
            print("{:6.2f} %".format(num_iters * 100 / len(train_loader)), end = "\r")

    # Save the average epoch loss
    avg_train_loss = np.mean(training_losses)
    med_train_loss = np.median(training_losses)
    net.log("Median training loss: {}".format(med_train_loss))
    net.log("Average training loss: {}".format(avg_train_loss))

    # Evaluate on the validation set
    # avg_loss_valid = evaluate(net, criterion, valid_loader, device)
    # y_loss_valid.append(avg_loss_valid)
    # logger.log("Average validation loss: {}".format(avg_loss_valid))

    #x_error_median, q_error_median, med_valid_loss = evaluate_median(net, criterion, itertools.islice(valid_loader, 10), device)
    x_error_median, q_error_median, med_valid_loss = evaluate_median(net, criterion, valid_loader, device)
    # y_loss_valid.append(loss_median)
    # y_training.append(np.median(training_loss))
    net.log("Median validation error: {:.2f} m, {:.2f} °".format(x_error_median, q_error_median))
    net.log("Median validation loss: {}".format(med_valid_loss))

    # Print some stats
    elapsed_time = stopwatch_epoch.stop()
    net.log("Epoch time: {:0.2f} minutes".format(elapsed_time))
    # print_torch_cuda_mem_usage()
    total_epoch_time += elapsed_time
    avg_epoch_time = total_epoch_time / (epoch + 1)
    training_time_left = (EPOCHS - epoch - 1) * avg_epoch_time
    net.log("Training time left: ~ {:.2f} minutes ({:.2f} hours)".format(training_time_left, training_time_left / 60))

    # # Plot the average loss over the epochs
    # loss_fig = plt.figure()
    # loss_ax = loss_fig.gca()
    # loss_ax.plot(x, y_loss_valid, "r", label = "Validation")
    # loss_ax.plot(x, y_training, "b", label = "Training")
    # plt.xlabel("Epoch")
    # plt.ylabel("Median loss")
    # loss_ax.grid(True)
    # loss_fig.legend()

    # loss_fig.savefig("{}/{}.loss.png".format(directory, identifier))
    # plt.close(loss_fig)

    # Save the model
    net.update_loss(med_train_loss, med_valid_loss)
    net.save()

# Elapsed training time
net.log("-----------------")
elapsed_time = stopwatch_train.stop()
net.log("Training time: {:0.2f} minutes".format(elapsed_time))
net.close()

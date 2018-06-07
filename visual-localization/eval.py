#!/usr/bin/env python3

import torch
import argparse

from dataset import DeepLocAugmented, make_loader, evaluate, evaluate_median
from network import PoseNet
from customized_loss import Customized_Loss
from preprocessing import validation_preprocessing

# Parse CLI arguments
args_parser = argparse.ArgumentParser(
	formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

args_parser.add_argument(
	"model_path",
	type = str,
	help = "Path to the saved model parameters file"
)

args_parser.add_argument(
	"--mode",
	type = str,
	help = "Dataset split to load (train / validation / test)",
	default = "validation"
)

args = args_parser.parse_args()

# Parameters
LOSS_BETA = int(args.model_path.split("_")[1])
MODE = args.mode
print("Beta: {}".format(LOSS_BETA))

# Device - use CPU is CUDA is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the saved parameters
net = PoseNet()
net.load_state_dict(torch.load(args.model_path))
net.to(device = device)

# Dataset
data = DeepLocAugmented(MODE, preprocess = validation_preprocessing)
print("{} samples: {}".format(MODE, len(data)))

# Loader
loader = make_loader(data, batch_size = 1)

# Loss function
criterion = Customized_Loss(beta = LOSS_BETA)

# Evaluate on the data
print("[{}]".format(MODE))
avg_loss = evaluate(net, criterion, loader, device)
print("Average {} loss: {}".format(MODE, avg_loss))
x_error_median, q_error_median = evaluate_median(net, loader, device)
print("Median {} error: {:.2f} m, {:.2f} °".format(MODE, x_error_median, q_error_median))

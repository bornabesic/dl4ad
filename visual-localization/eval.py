#!/usr/bin/env python3

import torch
import argparse

from dataset import DeepLocAugmented, make_loader, evaluate_median
import network
from customized_loss import Customized_Loss

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
	"architecture",
	type = str,
	help = "Neural network architecture to use",
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
ARCHITECTURE = args.architecture
print("Beta: {}".format(LOSS_BETA))

# Device - use CPU is CUDA is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the saved parameters
arch_class = getattr(network, ARCHITECTURE)
net = arch_class()
net.load_state_dict(torch.load(args.model_path))
net.to(device = device)

# Dataset
data = DeepLocAugmented(MODE, preprocess = None)
print("{} samples: {}".format(MODE.title(), len(data)))

# Loss function
criterion = Customized_Loss(beta = LOSS_BETA)

# Evaluate on the data
x_error_median, q_error_median, loss_median = evaluate_median(net, criterion, data, device)
print("Median {} error: {:.2f} m, {:.2f} °".format(MODE, x_error_median, q_error_median))
print("Median {} loss: {}".format(MODE, loss_median))

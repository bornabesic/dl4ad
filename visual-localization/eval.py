#!/usr/bin/env python3

import torch
import argparse

from dataset import DeepLocAugmented, make_loader, evaluate, evaluate_median
from network import PoseNet
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

args = args_parser.parse_args()

# Parameters
LOSS_BETA = int(args.model_path.split("_")[1])
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
test_data = DeepLocAugmented("test", preprocess = validation_preprocessing)
print("Test samples:", len(test_data))

# Loader
test_loader = make_loader(test_data, batch_size = 1)

# Loss function
criterion = Customized_Loss(beta = LOSS_BETA)

# TODO Evaluate on the test set
print("[TEST]")
avg_loss_test = evaluate(net, criterion, test_loader, device)
print("Average test loss: {}".format(avg_loss_test))
x_error_median, q_error_median = evaluate_median(net, test_loader, device)
print("Median test error: {:.2f} m, {:.2f} °".format(x_error_median, q_error_median))

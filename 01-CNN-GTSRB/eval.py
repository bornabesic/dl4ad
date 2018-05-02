#!/usr/bin/python3

import torch
import argparse

from dataset import GTSRBTest, make_test_loader, evaluate
from network import NeuralNetwork
from transform import shared_transform

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

# Load the saved parameters
net = NeuralNetwork()
net.load_state_dict(torch.load(args.model_path))
net.cuda()

# Dataset
test_data = GTSRBTest(
    test_path = "test",
    transform = shared_transform
)

print("Test samples:", len(test_data))

# Loader
test_loader = make_test_loader(test_data)

# Evaluate on the test set
print("[TEST]")
accuracy = evaluate(net, test_loader)
print("Test accuracy: {}%".format(accuracy))
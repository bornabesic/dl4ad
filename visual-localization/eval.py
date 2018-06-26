#!/usr/bin/env python3

import torch
import argparse
import numpy as np

import dataset
from dataset import DeepLocAugmented, make_loader, evaluate_median
import network
from customized_loss import Customized_Loss
from transformations import euler_from_quaternion

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
	"dataset",
	type = str,
	help = "Dataset to use",
)

args_parser.add_argument(
	"--mode",
	type = str,
	help = "Dataset split to load (train / validation / test)",
	default = "validation"
)

args_parser.add_argument(
	"--visualize",
	type = bool,
	help = "Visualize the ground truth and the network output",
	default = True
)

args = args_parser.parse_args()

# Parameters
LOSS_BETA = float(args.model_path.split("_")[1])
MODE = args.mode
DATASET = args.dataset
ARCHITECTURE = args.architecture
VISUALIZE = args.visualize
print("Beta: {}".format(LOSS_BETA))
print("Mode: {}".format(MODE))
print("Architecture: {}".format(ARCHITECTURE))
print("Dataset: {}".format(DATASET))

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
data_class = getattr(dataset, DATASET)
data = data_class(MODE, preprocess = data_class.valid_preprocessing)
data_loader = make_loader(data, batch_size = 1, num_workers = 4)
print("Samples: {}".format(len(data)))

# Loss function
criterion = Customized_Loss(beta = LOSS_BETA)

# Evaluate on the data
x_error_median, q_error_median, loss_median = evaluate_median(net, criterion, data_loader, device)
print("Median {} error: {:.2f} m, {:.2f} °".format(MODE, x_error_median, q_error_median))
print("Median {} loss: {}".format(MODE, loss_median))


if VISUALIZE:
	from visualize import PosePlotter

	plotter = PosePlotter()
	plotter.register("Ground truth", "red")
	plotter.register("NN prediction", "blue")
	net.eval()
	for images, ps in data_loader:
		images = images.to(device = device)
		ps = ps.to(device = device)
		
		ps_outs = net(images)
		# Prediction
		x_pred, y_pred, qw_pred, qx_pred, qy_pred, qz_pred = p_out = ps_outs[-1].view(-1).cpu().detach().numpy()
		x_pred, y_pred, _ = np.array([x_pred, y_pred, 0]) + data.origin
		lat_pred, lng_pred = PosePlotter.utm2latlng(x_pred, y_pred)
		_, _, theta_pred = euler_from_quaternion([qw_pred, qx_pred, qy_pred, qz_pred])
		# Ground truth
		x, y, qw, qx, qy, qz = ps.view(-1).cpu().numpy()
		x, y, _ = np.array([x, y, 0]) + data.origin
		lat, lng = PosePlotter.utm2latlng(x, y)
		_, _, theta = euler_from_quaternion([qw, qx, qy, qz])
		# Visualize on the map
		plotter.update("Ground truth", x, y, theta)
		plotter.update("NN prediction", x_pred, y_pred, theta_pred)
		plotter.draw()

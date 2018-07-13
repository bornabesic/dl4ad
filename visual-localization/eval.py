#!/usr/bin/env python3

import torch
import argparse
import numpy as np

import dataset
from dataset import PerceptionCarDataset, PerceptionCarDatasetMerged, make_loader, evaluate_median, meters_and_degrees_error
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
        "--only_front_camera",
        type = bool,
        help = "Use only front camera",
        default = False
)

args_parser.add_argument(
        "--update_interval",
        type = float,
        help = "Time per image/pose",
        default = 0.5
)

args_parser.add_argument(
	"--mode",
	type = str,
	help = "Dataset split to load (train / validation / test)",
	default = "validation"
)


args_parser.add_argument(
        "--split",
        type = str,
        help = "Split mode. manual(default) or traj)",
        default = "manual"
)


args_parser.add_argument(
	"--visualize",
	type = bool,
	help = "Visualize the ground truth and the network output",
	default = True
)

args = args_parser.parse_args()

# Parameters
MODEL_PATH = args.model_path
MODE = args.mode
ONLY_FRONT_CAMERA = args.only_front_camera
SPLITMODE = args.split
# DATASET = args.dataset
VISUALIZE = args.visualize
UPDATE_INTERVAL = args.update_interval
print("Mode: {}".format(MODE))

# Device - use CPU is CUDA is not available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load the saved parameters
net = network.NeuralNetworkModel.load(MODEL_PATH, device)
net.to(device = device)

# Dataset
data = PerceptionCarDatasetMerged("PerceptionCarDataset","PerceptionCarDataset2", mode = MODE,split = SPLITMODE, only_front_camera = ONLY_FRONT_CAMERA, preprocess = PerceptionCarDataset.valid_preprocessing)
data_loader = make_loader(data, shuffle = False, batch_size = 1, num_workers = 4)
print("Samples: {}".format(len(data)))

# Loss function
criterion = Customized_Loss()

if VISUALIZE:
        from visualize import PosePlotter
        plotter = PosePlotter(update_interval = UPDATE_INTERVAL, trajectory = False)
        plotter.register("Ground truth", "red")
        plotter.register("NN prediction", "blue")
        net.eval()

        for images, ps in data_loader:
                images = images.to(device = device)
                ps = ps.to(device = device)
                ps_outs = net(images)
                # Prediction
                x_pred, y_pred, cosine, sine = ps_outs[-1].view(-1).cpu().detach().numpy()
                theta_pred = np.arctan2(sine, cosine)
                x_pred, y_pred, theta_pred = PerceptionCarDataset.unnormalize(x_pred, y_pred, theta_pred)
                lat_pred, lng_pred = PosePlotter.utm2latlng(x_pred, y_pred)
		# Ground truth
                x, y, cosine, sine = ps.view(-1).cpu().numpy()
                theta = np.arctan2(sine, cosine)
                x, y, theta = PerceptionCarDataset.unnormalize(x, y, theta)
                lat, lng = PosePlotter.utm2latlng(x, y)
                # Print the error
                position_error, orientation_error = meters_and_degrees_error(np.array([x, y]), theta, np.array([x_pred, y_pred]), theta_pred)
                print("Error: {:.2f} m, {:.2f} °".format(position_error, orientation_error))
		# Visualize on the map
                plotter.update("Ground truth", x, y, theta)
                plotter.update("NN prediction", x_pred, y_pred, theta_pred)
                plotter.draw()
else:
        # Evaluate on the data
        x_error_median, q_error_median, loss_median = evaluate_median(net, criterion, data_loader, device)
        print("Median {} error: {:.2f} m, {:.2f} °".format(MODE, x_error_median, q_error_median))
        print("Median {} loss: {}".format(MODE, loss_median))
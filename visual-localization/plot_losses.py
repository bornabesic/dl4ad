import json
import argparse
import os
import matplotlib as plt

# Parse CLI arguments
args_parser = argparse.ArgumentParser(
	formatter_class = argparse.ArgumentDefaultsHelpFormatter
)

args_parser.add_argument(
	"model_path",
	type = str,
	help = "Path to model that should be plotted"
)
args = args_parser.parse_args()

# Split model path
MODEL_PATH = args.model_path
path, filename = os.path.split(MODEL_PATH)
intentifier, _ = os.path.splitext(filename)

# Read model information
with open(MODEL_PATH, "rt") as json_file:
    params = json.load(json_file)

architecture = params["architecture"]
model_path = params["model_path"]
train_losses = params["train_losses"]
valid_losses = params["valid_losses"]
only_front_camera = params["only_front_camera"]

# Plot the losses over the epochs
loss_fig = plt.figure()
loss_ax = loss_fig.gca()
loss_ax.plot(x, valid_losses, "r", label = "Validation")
loss_ax.plot(x, train_losses, "b", label = "Training")
plt.xlabel("Epoch")
plt.ylabel("Median loss")
loss_ax.grid(True)
if only_front_camera:
    plt.title(architecture + " with 1 image")
else:
    plt.title(architecture + "with 6 images")
loss_fig.legend()
loss_fig.savefig("{}/{}.loss.png".format(path, identifier))
plt.close(losse_fig)
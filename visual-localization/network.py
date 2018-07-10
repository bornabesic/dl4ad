import os
import json
import datetime
import torch
import torch.nn as nn
from utils import Logger

import sys
this = sys.modules[__name__]

from modules import Flatten, Inception

def parameters(model):
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        trainable += (p.numel() if p.requires_grad else 0)
    return (trainable, total)

def xavier_initialization(module):
    try:
        nn.init.xavier_uniform_(module.weight)
    except AttributeError:
        pass

def get_model_class(model_name):
    return getattr(this, model_name)    

class NeuralNetworkModel(nn.Module):

    directory = "models"

    def __init__(self, architecture, **kwargs):
        super(NeuralNetworkModel, self).__init__()
        
        self.architecture = architecture
        self.kwargs = kwargs

        # Identifier
        timestamp = datetime.datetime.now()
        self.identifier = "{}_{}_{}_{}_{}_{}".format(architecture, timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)

        # File paths
        base_path = os.path.join(NeuralNetworkModel.directory, self.identifier)
        self.model_path = base_path + ".pth"
        self.json_path = base_path + ".json"

        # Logger
        self.logger = Logger(base_path + ".log", print_to_stdout = True)

        # State
        self.train_losses = list()
        self.valid_losses = list()

    def update_loss(self, train_loss, valid_loss):
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)

    def save(self):
        os.makedirs(NeuralNetworkModel.directory, exist_ok = True)
        with open(self.json_path, "wt") as json_file:
            json.dump(
                {
                    "architecture": self.architecture,
                    "train_losses": self.train_losses,
                    "valid_losses": self.valid_losses,
                    "model_path": self.model_path,
                    **self.kwargs
                },
                json_file
            )

        torch.save(self.state_dict(), self.model_path)
        self.log("Model parameters saved to {}.".format(self.model_path))

    def log(self, text):
        self.logger.log(text)

    def close(self):
        self.logger.close()

    @staticmethod
    def load(json_path):
        with open(json_path, "rt") as json_file:
            params = json.load(json_file)

        architecture = params["architecture"]
        model_path = params["model_path"]
        train_losses = params["train_losses"]
        valid_losses = params["valid_losses"]

        del params["architecture"]
        del params["model_path"]
        del params["train_losses"]
        del params["valid_losses"]

        arch_class = get_model_class(architecture)
        net = arch_class(**params)
        net.load_state_dict(torch.load(model_path))
        net.train_losses = train_losses
        net.valid_losses = valid_losses

        return net


# Modified implementation originally taken from:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py

class PoseNet(NeuralNetworkModel):

    def __init__(self, **kwargs):
        super(PoseNet, self).__init__("PoseNet", **kwargs)

        # Stem network
        self.stem_network = nn.Sequential(
            nn.Conv2d(in_channels = 18, out_channels = 64, kernel_size = 7, stride = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.ReLU(True),
            #
            nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, stride = 1),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.ReLU(True),
        )

        # Side networks
        self.side_network_4a = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 1, stride = 1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(3 * 3 * 128, 1024), # paper says 4 x 4 ?
            nn.ReLU(True),
            nn.Dropout(p = 0.7),
            nn.Linear(1024, 4)
        )

        self.side_network_4d = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            nn.Conv2d(in_channels = 528, out_channels = 128, kernel_size = 1, stride = 1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(3 * 3 * 128, 1024), # paper says 4 x 4 ?
            nn.ReLU(True),
            nn.Dropout(p = 0.7),
            nn.Linear(1024, 4)
        )

        # Inceptions 3
        self.incep_3a = Inception(
            in_channels = 192,
            conv1x1_out_channels = 64,
            conv3x3_in_channels = 96, conv3x3_out_channels = 128,
            conv5x5_in_channels = 16, conv5x5_out_channels = 32,
            maxpool3x3_out_channels = 32,
        )

        self.incep_3b = Inception(
            in_channels = 256,
            conv1x1_out_channels = 128,
            conv3x3_in_channels = 128, conv3x3_out_channels = 192,
            conv5x5_in_channels = 32, conv5x5_out_channels = 96,
            maxpool3x3_out_channels = 64,
        )

        # Inceptions 4
        self.incep_4a = Inception(
            in_channels = 480,
            conv1x1_out_channels = 192,
            conv3x3_in_channels = 96, conv3x3_out_channels = 208,
            conv5x5_in_channels = 16, conv5x5_out_channels = 48,
            maxpool3x3_out_channels = 64
        )
        self.incep_4b = Inception(
            in_channels = 512,
            conv1x1_out_channels = 160,
            conv3x3_in_channels = 112, conv3x3_out_channels = 224,
            conv5x5_in_channels = 24, conv5x5_out_channels = 64,
            maxpool3x3_out_channels = 64
        )

        self.incep_4c = Inception(
            in_channels = 512,
            conv1x1_out_channels = 128,
            conv3x3_in_channels = 128, conv3x3_out_channels = 256,
            conv5x5_in_channels = 24,  conv5x5_out_channels = 64,
            maxpool3x3_out_channels = 64
        )
        self.incep_4d = Inception(
            in_channels = 512,
            conv1x1_out_channels = 112,
            conv3x3_in_channels = 144, conv3x3_out_channels = 288,
            conv5x5_in_channels = 32, conv5x5_out_channels = 64,
            maxpool3x3_out_channels = 64
        )
        self.incep_4e = Inception(
            in_channels = 528,
            conv1x1_out_channels = 256,
            conv3x3_in_channels = 160, conv3x3_out_channels = 320,
            conv5x5_in_channels = 32, conv5x5_out_channels = 128,
            maxpool3x3_out_channels = 128
        )

        # Inceptions 5
        self.incep_5a = Inception(
            in_channels = 832,
            conv1x1_out_channels = 256,
            conv3x3_in_channels = 160, conv3x3_out_channels = 320,
            conv5x5_in_channels = 32, conv5x5_out_channels = 128,
            maxpool3x3_out_channels = 128
        )
        self.incep_5b = Inception(
            in_channels = 832,
            conv1x1_out_channels = 384,
            conv3x3_in_channels = 192, conv3x3_out_channels = 384,
            conv5x5_in_channels = 48, conv5x5_out_channels = 128,
            maxpool3x3_out_channels = 128
        )

        self.flatten = Flatten()
        self.dropout = nn.Dropout(p = 0.4)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        
        self.final_regressor = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4)
        )

        self.apply(xavier_initialization)

    def forward(self, x):
        out1 = None
        out2 = None

        out = self.stem_network(x)

        out = self.incep_3a(out)
        out = self.incep_3b(out)
        out = self.maxpool(out)

        out = self.incep_4a(out)
        if self.training:
            out1 = self.side_network_4a(out)
        out = self.incep_4b(out)
        out = self.incep_4c(out)
        out = self.incep_4d(out)
        if self.training:
            out2 = self.side_network_4d(out)
        out = self.incep_4e(out)
        out = self.maxpool(out)

        out = self.incep_5a(out)
        out = self.incep_5b(out)
        out = self.avgpool(out)

        out = self.flatten(out)
        out = self.dropout(out)
        out3 = self.final_regressor(out)
        return (out1, out2, out3)

class PoseNetSimple(NeuralNetworkModel):

    def __init__(self, **kwargs):
        super(PoseNetSimple, self).__init__("PoseNetSimple", **kwargs)

        if kwargs["only_front_camera"]:
            in_channels = 3
        else:
            in_channels = 18

        # Stem network
        self.stem_network = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 7, stride = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.LeakyReLU(inplace = True),
            #
            nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, stride = 1),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.LeakyReLU(inplace = True),
        )

        # Inceptions 3
        self.incep_3a = Inception(
            in_channels = 192,
            conv1x1_out_channels = 64,
            conv3x3_in_channels = 96, conv3x3_out_channels = 128,
            conv5x5_in_channels = 16, conv5x5_out_channels = 32,
            maxpool3x3_out_channels = 32,
        )

        self.incep_3b = Inception(
            in_channels = 256,
            conv1x1_out_channels = 128,
            conv3x3_in_channels = 128, conv3x3_out_channels = 192,
            conv5x5_in_channels = 32, conv5x5_out_channels = 96,
            maxpool3x3_out_channels = 64,
        )

        # Inceptions 4
        self.incep_4a = Inception(
            in_channels = 480,
            conv1x1_out_channels = 192,
            conv3x3_in_channels = 96, conv3x3_out_channels = 208,
            conv5x5_in_channels = 16, conv5x5_out_channels = 48,
            maxpool3x3_out_channels = 64
        )

        # Side networks
        self.side_network_4a = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 1, stride = 1),
            nn.LeakyReLU(inplace = True),
            Flatten(),
            nn.Linear(3 * 3 * 128, 1024), # paper says 4 x 4 ?
            nn.LeakyReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(1024, 4)
        )

        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.apply(xavier_initialization)

    def forward(self, x):
        out_stem = self.stem_network(x)

        out_incep_3a = self.incep_3a(out_stem)
        out_incep_3b = self.incep_3b(out_incep_3a)
        out_maxpool = self.maxpool(out_incep_3b)

        out_incep_4a = self.incep_4a(out_maxpool)
        out1 = self.side_network_4a(out_incep_4a)
        assert not torch.isnan(out1).byte().any() # All elements in the tensor are zero (no NaNs)
        assert not (out1 == float("inf")).byte().any() # All elements in the tensor are zero (no infs)
        return (out1,)

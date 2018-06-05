import torch
import torch.nn as nn

from modules import Flatten, Inception

def parameters(model):
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        trainable += (p.numel() if p.requires_grad else 0)
    return (trainable, total)

# Modified implementation originally taken from:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py

class PoseNet(nn.Module):

    def __init__(self):
        super(PoseNet, self).__init__()

        # Stem network
        self.stem_network = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2),
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
            nn.Linear(1024, 7)
        )

        self.side_network_4d = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            nn.Conv2d(in_channels = 528, out_channels = 128, kernel_size = 1, stride = 1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(3 * 3 * 128, 1024), # paper says 4 x 4 ?
            nn.ReLU(True),
            nn.Dropout(p = 0.7),
            nn.Linear(1024, 7)
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
            nn.Linear(2048, 7)
        )

    def forward(self, x):
        out = self.stem_network(x)

        out = self.incep_3a(out)
        out = self.incep_3b(out)
        out = self.maxpool(out)

        out = self.incep_4a(out)
        out1 = self.side_network_4a(out)
        out = self.incep_4b(out)
        out = self.incep_4c(out)
        out = self.incep_4d(out)
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

class PoseNetSimple(nn.Module):

    def __init__(self):
        super(PoseNetSimple, self).__init__()

        # Stem network
        self.stem_network = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2),
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
            nn.Linear(1024, 7)
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

        self.flatten = Flatten()
        self.dropout = nn.Dropout(p = 0.4)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)

    def forward(self, x):
        out = self.stem_network(x)

        out = self.incep_3a(out)
        out = self.incep_3b(out)
        out = self.maxpool(out)

        out = self.incep_4a(out)
        out1 = self.side_network_4a(out)
        return out1
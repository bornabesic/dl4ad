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

        self.stem_network = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.ReLU(True),
            #
            nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3, stride = 1),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.ReLU(True),
        )

        self.side_network_a4 = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 1, stride = 1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(3 * 3 * 128, 1024), # paper says 4 x 4 ?
            nn.ReLU(True),
            nn.Dropout(p = 0.7),
            nn.Linear(1024, 7)
        )

        self.side_network_d4 = nn.Sequential(
            nn.AvgPool2d(kernel_size = 5, stride = 3),
            nn.Conv2d(in_channels = 528, out_channels = 128, kernel_size = 1, stride = 1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(3 * 3 * 128, 1024), # paper says 4 x 4 ?
            nn.ReLU(True),
            nn.Dropout(p = 0.7),
            nn.Linear(1024, 7)
        )

        self.a3 = Inception(
            in_channels = 192,
            conv1x1_out_channels = 64,
            conv3x3_in_channels = 96, conv3x3_out_channels = 128,
            conv5x5_in_channels = 16, conv5x5_out_channels = 32,
            maxpool3x3_out_channels = 32,
        )
            
        self.b3 = Inception(
            in_channels = 256,
            conv1x1_out_channels = 128,
            conv3x3_in_channels = 128, conv3x3_out_channels = 192,
            conv5x5_in_channels = 32, conv5x5_out_channels = 96,
            maxpool3x3_out_channels = 64,
        )

        self.a4 = Inception(
            in_channels = 480,
            conv1x1_out_channels = 192,
            conv3x3_in_channels = 96, conv3x3_out_channels = 208,
            conv5x5_in_channels = 16, conv5x5_out_channels = 48,
            maxpool3x3_out_channels = 64
        )
        self.b4 = Inception(
            in_channels = 512,
            conv1x1_out_channels = 160,
            conv3x3_in_channels = 112, conv3x3_out_channels = 224,
            conv5x5_in_channels = 24, conv5x5_out_channels = 64,
            maxpool3x3_out_channels = 64
        )

        self.c4 = Inception(
            in_channels = 512,
            conv1x1_out_channels = 128,
            conv3x3_in_channels = 128, conv3x3_out_channels = 256,
            conv5x5_in_channels = 24,  conv5x5_out_channels = 64,
            maxpool3x3_out_channels = 64
        )
        self.d4 = Inception(
            in_channels = 512,
            conv1x1_out_channels = 112,
            conv3x3_in_channels = 144, conv3x3_out_channels = 288,
            conv5x5_in_channels = 32, conv5x5_out_channels = 64,
            maxpool3x3_out_channels = 64
        )
        self.e4 = Inception(
            in_channels = 528,
            conv1x1_out_channels = 256,
            conv3x3_in_channels = 160, conv3x3_out_channels = 320,
            conv5x5_in_channels = 32, conv5x5_out_channels = 128,
            maxpool3x3_out_channels = 128
        )

        self.a5 = Inception(
            in_channels = 832,
            conv1x1_out_channels = 256,
            conv3x3_in_channels = 160, conv3x3_out_channels = 320,
            conv5x5_in_channels = 32, conv5x5_out_channels = 128,
            maxpool3x3_out_channels = 128
        )
        self.b5 = Inception(
            in_channels = 832,
            conv1x1_out_channels = 384,
            conv3x3_in_channels = 192, conv3x3_out_channels = 384,
            conv5x5_in_channels = 48, conv5x5_out_channels = 128,
            maxpool3x3_out_channels = 128
        )

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

        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)

        out = self.a4(out)
        out1 = self.side_network_a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out2 = self.side_network_d4(out)
        out = self.e4(out)
        out = self.maxpool(out)

        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out3 = self.final_regressor(out)
        return (out1, out2, out3)

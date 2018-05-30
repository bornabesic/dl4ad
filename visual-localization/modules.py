import torch
import torch.nn as nn

# Modified implementation originally taken from:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py

class Inception(nn.Module):

    def __init__(self,
            in_channels,
            # Branch 1
            conv1x1_out_channels,
            # Branch 2
            conv3x3_in_channels,
            conv3x3_out_channels,
            # Branch 3
            conv5x5_in_channels,
            conv5x5_out_channels,
            # Branch 4
            maxpool3x3_out_channels,
        ):
        super(Inception, self).__init__()

        # 1 x 1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, conv1x1_out_channels, kernel_size = 1),
            # nn.BatchNorm2d(conv1x1_out_channels),
            nn.ReLU(True),
        )

        # 1 x 1 conv -> 3 x 3 conv branch
        self.branch2 = nn.Sequential(
            # 1 x 1
            nn.Conv2d(in_channels, conv3x3_in_channels, kernel_size = 1),
            nn.BatchNorm2d(conv3x3_in_channels),
            nn.ReLU(True),
            # 3 x 3
            nn.Conv2d(conv3x3_in_channels, conv3x3_out_channels, kernel_size = 3, padding = 1),
            # nn.BatchNorm2d(conv3x3_out_channels),
            nn.ReLU(True),
        )

        # 1 x 1 conv -> 5 x 5 conv branch
        self.branch3 = nn.Sequential(
            # 1 x 1
            nn.Conv2d(in_channels, conv5x5_in_channels, kernel_size = 1, padding = 1),
            nn.BatchNorm2d(conv5x5_in_channels),
            nn.ReLU(True),
            # 5 x 5
            nn.Conv2d(conv5x5_in_channels, conv5x5_out_channels, kernel_size = 5, padding = 1),
            # nn.BatchNorm2d(conv5x5_out_channels),
            nn.ReLU(True)
        )

        # 3 x 3 pool -> 1 x 1 conv branch
        self.branch4 = nn.Sequential(
            # pool
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            # 1 x 1
            nn.Conv2d(in_channels, maxpool3x3_out_channels, kernel_size = 1),
            # nn.BatchNorm2d(maxpool3x3_out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], 1)
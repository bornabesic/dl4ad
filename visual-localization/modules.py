import torch
import torch.nn as nn

# Taken from:
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py

class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()

        # 1 x 1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size = 1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1 x 1 conv -> 3 x 3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size = 1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1 x 1 conv -> 5 x 5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size = 1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3 x 3 pool -> 1 x 1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride = 1, padding = 1),
            nn.Conv2d(in_planes, pool_planes, kernel_size = 1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        return torch.cat([y1, y2, y3, y4], 1)
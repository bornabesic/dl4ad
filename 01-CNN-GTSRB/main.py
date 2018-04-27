#!/usr/bin/python3

import torch
from torchvision import transforms

from dataset import GTSRB
from preprocessing import Resize, SubtractMean, MakeTensor

data = GTSRB(
        training_path = "training",
        transform = transforms.Compose([
            Resize(32, 32),
            SubtractMean(),
            MakeTensor()
        ]
    ))


#!/usr/bin/env python3

import torch

from dataset import DeepLoc

train_data = DeepLoc("train")
print("Training set size: {} samples".format(len(train_data)))


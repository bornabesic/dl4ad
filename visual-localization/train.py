#!/usr/bin/env python3

import torch

from dataset import DeepLocAugmented, make_train_valid_loader, make_test_loader

train_data = DeepLocAugmented("train")
print("Training set size: {} samples".format(len(train_data)))

train_loader, valid_loader = make_train_valid_loader(train_data, valid_percentage = 0.2)

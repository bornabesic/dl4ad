#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import GTSRB
from preprocessing import Resize, SubtractMean, MakeTensor
from network import NeuralNetwork

data = GTSRB(
    training_path = "training",
    transform = transforms.Compose([
        Resize(32, 32),
        SubtractMean(),
        MakeTensor()
    ]
))

dataloader = DataLoader(data,
    batch_size = 4,
    shuffle = True,
    num_workers = 4
)

net = NeuralNetwork()
net.cuda()

cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 1e-3, momentum = 0.9)

for i, batch in enumerate(dataloader):
    Xs, ys = batch
    Xs, ys = Xs.cuda(), ys.cuda()
    optimizer.zero_grad()

    # Forward pass
    ys_hat = net(Xs)
    loss = cost(ys_hat, ys)
    print("Loss: ", loss)

    input()
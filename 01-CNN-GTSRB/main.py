#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

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
optimizer = optim.SGD(net.parameters(), lr = 1e-3)

for epoch in range(10):
    total_loss = 0
    print("[Epoch {}]".format(epoch))
    for i, batch in enumerate(dataloader):
        Xs, ys = batch
        Xs, ys = Variable(Xs.cuda()), Variable(ys.cuda())

        # Forward pass
        ys_hat = net(Xs)
        loss = cost(ys_hat, ys)
        total_loss += loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimize
        optimizer.step()
    print("Average loss: {}".format(total_loss / len(dataloader)))
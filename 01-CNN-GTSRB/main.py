#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from dataset import GTSRBTraining, GTSRBTest, make_train_valid_loader, make_test_loader, evaluate
from preprocessing import Resize, SubtractMean, MakeTensor
from network import NeuralNetwork

NUM_EPOCHS = 15

# Transformations
shared_transform = transforms.Compose([
    Resize(32, 32),
    SubtractMean(),
    MakeTensor()
])

# Dataset
training_data = GTSRBTraining(
    training_path = "training",
    transform = shared_transform
)

test_data = GTSRBTest(
    test_path = "test",
    transform = shared_transform
)

print("Training samples:", len(training_data))
print("Test samples:", len(test_data))

# Loaders
train_loader, valid_loader = make_train_valid_loader(training_data, valid_percentage = 0.2)
test_loader = make_test_loader(test_data)

# Model
net = NeuralNetwork()
net.cuda()

# Loss function and optimization method
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 1e-4, momentum = 0.5)

# Plot variables
plt.hold(True)
x = []
y_validation = []
y_training = []

# Epochs
for epoch in range(NUM_EPOCHS):
    x.append(epoch + 1)
    print("[Epoch {}]".format(epoch + 1))

    # Train
    net.train()
    for batch_idx, (Xs, ys) in enumerate(train_loader):
        Xs, ys = Xs.cuda(), ys.cuda()

        # Forward pass
        ys_hat = net(Xs)
        loss = cost(ys_hat, ys)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimize
        optimizer.step()

        # Report loss every 50 batches
        if batch_idx % 50 == 0:
            print("{} / {} => loss = {}".format(batch_idx * len(Xs), len(train_loader.dataset), loss.item()))

    # Evaluate on the training set
    accuracy = evaluate(net, train_loader)
    y_training.append(accuracy)
    print("Training accuracy: {}%".format(accuracy))

    # Evaluate on the validation set
    accuracy = evaluate(net, valid_loader)
    y_validation.append(accuracy)
    print("Validation accuracy: {}%".format(accuracy))

# Plot validation accuracy over epochs
plt.plot(x, y_training, "b", label = "Training")
plt.plot(x, y_validation, "r", label = "Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.savefig("learning_curve.png")

# Evaluate on the test set
print("[TEST]")
accuracy = evaluate(net, test_loader)
print("Test accuracy: {}%".format(accuracy))

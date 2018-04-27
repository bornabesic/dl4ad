
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3) # 64 filters: 3 x 3
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3) # 128 filters: 3 x 3
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3) # 128 filters: 3 x 3
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3) # 128 filters: 3 x 3
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc = nn.Linear(128 * 5 * 5, 43)

    def forward(self, x):
        # Conv, ReLU, Conv, ReLU, Max-pooling
        relu1 = F.relu(self.conv1(x.float()))
        relu2 = F.relu(self.conv2(relu1))
        mp1 = self.maxpool1(relu2)

        # Conv, ReLU, Conv, ReLU, Max-pooling
        relu3 = F.relu(self.conv3(mp1))
        relu4 = F.relu(self.conv4(relu3))
        mp2 = self.maxpool2(relu4)

        # Fully-connected layer
        flat = mp2.view(mp2.size(0), -1)
        y = F.softmax(self.fc(flat))

        return y


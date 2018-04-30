
import os
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import io

from utils import lines

def read_annotation(annotation_path):
    data = []
    for line in lines(annotation_path):
        cells = line.split(";")
        if cells[0] == "Filename":
            continue
        image_path, width, height, klass = os.path.join(os.path.dirname(annotation_path), cells[0]), int(cells[1]), int(cells[2]), int(cells[-1])
        data.append((image_path, width, height, klass))
    return data

def make_train_valid_loader(data, valid_percentage, batch_size = 4, num_workers = 1, pin_memory = True):
    num_samples = len(data)
    indices = list(range(num_samples))
    split = int(valid_percentage * num_samples)

    np.random.shuffle(indices)

    train_idxs, valid_idxs = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idxs)
    valid_sampler = SubsetRandomSampler(valid_idxs)

    train_loader = DataLoader(data,
        batch_size = batch_size,
        sampler = train_sampler,
        num_workers = num_workers,
        pin_memory = pin_memory
    )
    valid_loader = DataLoader(data,
        batch_size = batch_size,
        sampler = valid_sampler,
        num_workers = num_workers,
        pin_memory = pin_memory
    )

    return train_loader, valid_loader

def make_test_loader(data, batch_size = 4, shuffle = True, num_workers = 1, pin_memory = True):
    return DataLoader(data,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = shuffle
    )

def evaluate(model, loader):
    model.eval()
    total_predictions = 0
    correct_predictions = 0
    for Xs, ys in loader:
        Xs, ys = Xs.cuda(), ys.cuda()
        ys_hat = model(Xs)
        _, predicted = torch.max(ys_hat.data, 1)
        total_predictions += ys.size(0)
        correct_predictions += (predicted == ys.data).sum().item()
    accuracy = correct_predictions * 100.0 / total_predictions
    return accuracy

class GTSRB(Dataset):
    def __init__(self, transform = None):
        self.size = 0
        self.data = []
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path, width, height, klass = self.data[idx]
        sample = io.imread(image_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return (sample, klass)

class GTSRBTraining(GTSRB):
    def __init__(self, training_path, transform = None):
        self.transform = transform
        self.class_dirs = [o for o in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, o))]

        self.num_classes = len(self.class_dirs)

        self.data = []
        for d in self.class_dirs:
            csv_file_name = "GT-{}.csv".format(d)
            annotation_path = os.path.join(training_path, d, csv_file_name)
            self.data += read_annotation(annotation_path)

        self.size = len(self.data)

class GTSRBTest(GTSRB):
    def __init__(self, test_path, transform = None):
        self.transform = transform
        annotation_path = os.path.join(test_path, "GT-final_test.test.csv")
        self.data = read_annotation(annotation_path)
        self.size = len(self.data)


import os.path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image

from utils import read_table
from preprocessing import default_preprocessing

class DeepLoc(Dataset):

    def __init__(self, mode, preprocess = default_preprocessing):
        self.data = []
        self.preprocess = preprocess

        if mode not in ("train", "test"):
            raise ValueError("Only 'train' and 'test' modes are available.")

        mode_path = os.path.join("DeepLoc", mode)
        poses_path = os.path.join(mode_path, "poses.txt")

        for filename, x, y, z, qw, qx, qy, qz in read_table(
            poses_path,
            types = (str, float, float, float, float, float, float, float),
            delimiter = " "):
                pose = (x, y, z, qw, qx, qy, qz)
                image_path = os.path.join(mode_path, "LeftImages", "{}.png".format(filename))
                self.data.append((image_path, pose))

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path, pose = self.data[idx]
        image = Image.open(image_path)
        if self.preprocess is not None:
            image = self.preprocess(image)

        x, y, z, qw, qx, qy, qz = pose

        # x = torch.tensor([x, y, z]) # 3D camera location (vector)
        # q = torch.tensor([qw, qx, qy, qz]) # Rotation (quaternion)
        p = torch.Tensor([x, y, z, qw, qx, qy, qz])
        return (image, p)

class DeepLocAugmented(DeepLoc):

    def __init__(self, mode, preprocess = default_preprocessing):
        self.data = []
        self.preprocess = preprocess

        if mode not in ("train", "validation", "test"):
            raise ValueError("Only 'train', 'validation' and 'test' modes are available.")

        mode_path = os.path.join("DeepLocAugmented", mode)
        poses_path = os.path.join(mode_path, "poses.txt")

        for image_path, x, y, z, qw, qx, qy, qz in read_table(
            poses_path,
            types = (str, float, float, float, float, float, float, float),
            delimiter = "\t"):
                pose = (x, y, z, qw, qx, qy, qz)
                self.data.append((image_path, pose))

        self.size = len(self.data)

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

def make_train_valid_generator(data, valid_percentage):
    num_samples = len(data)
    indices = list(range(num_samples))
    split = int(valid_percentage * num_samples)

    np.random.shuffle(indices)

    train_idxs, valid_idxs = indices[split:], indices[:split]

    def generator(idxs):
        for i in idxs:
            yield data[i]

    return generator(train_idxs), generator(valid_idxs)

def make_test_loader(data, batch_size = 4, shuffle = True, num_workers = 1, pin_memory = True):
    return DataLoader(data,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = shuffle
    )

def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss = 0
    num_iters = 0
    for images, ps in loader:
        ps = ps.to(device = device)
        images = images.to(device = device)
        ps_out1, ps_out2, ps_out3 = model(images)
        loss3 = criterion(ps_out3, ps)
        total_loss += loss3.item() # Important to use .item() !
        num_iters += 1
    avg_loss = total_loss / num_iters
    return avg_loss

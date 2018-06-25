
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import torchvision.transforms as transforms

from utils import read_table, lines
from preprocessing import default_preprocessing, validation_resize, validation_crops, validation_tensor, validation_preprocessing

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

class PerceptionCarDataset(Dataset):
    def __init__(self, mode, preprocess = default_preprocessing):
        self.data = []
        self.preprocess = preprocess

        if mode not in ("train", "validation", "test", "visualize"):
            raise ValueError("Invalid mode.")

        self.mode = mode

        set_path = "PerceptionCarDataset"
        
        origin_path = os.path.join(set_path, "origin.txt")
        for line in lines(origin_path):
          self.origin = torch.Tensor(tuple(map(float, line.split(" "))))

        #camera_paths = [
         #   os.path.join("front", "center"),
          #  os.path.join("front", "left"),
         #   os.path.join("front", "right"),
          #  os.path.join("back", "center"),
           # os.path.join("back", "left"),
            #os.path.join("back", "right")
        #]

        #for camera_path in camera_paths:
         #   poses_path = os.path.join(mode_path, camera_path, "poses.txt")'''

        if mode == "visualize":
            mode_path = os.path.join(set_path, "front", "center")
            poses_path = os.path.join(mode_path, "poses.txt")
            for filename, x, y, qw, qx, qy, qz in read_table(
            poses_path,
            types = (str, float, float, float, float, float, float),
            delimiter = " "):
                pose = (x, y, qw, qx, qy, qz)
                image_path = os.path.join(mode_path, filename)
                self.data.append((image_path, pose))
        else:
            poses_path = os.path.join(set_path, mode + ".txt")
            for *filenames, x, y, qw, qx, qy, qz in read_table(
                poses_path,
                types = (str,str,str,str,str,str, float, float, float, float, float, float),
                delimiter = " "):
                    pose = (x, y, qw, qx, qy, qz)
                    image_paths = map(lambda fn: os.path.join(set_path, fn), filenames)
                    self.data.append((*image_paths, pose))

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.mode == "visualize":
            image_path, pose = self.data[idx]
            image = Image.open(image_path)
            if self.preprocess is not None:
                image = self.preprocess(image)
        else:
            *image_paths, pose = self.data[idx]

            images = map(Image.open, image_paths)

            if self.preprocess is not None:
                images = map(self.preprocess, images)

            # Concatinate all images (each 3 layer) to one image with 18 layers
            images = tuple(images)
            image = torch.cat(images, dim = 0)
        
        x, y, qw, qx, qy, qz = pose
        p = torch.Tensor([x, y, qw, qx, qy, qz])

        return (image, p)

# def make_train_valid_loader(data, valid_percentage, batch_size = 4, num_workers = 1, pin_memory = True):
#     num_samples = len(data)
#     indices = list(range(num_samples))
#     split = int(valid_percentage * num_samples)

#     np.random.shuffle(indices)

#     train_idxs, valid_idxs = indices[split:], indices[:split]
#     train_sampler = SubsetRandomSampler(train_idxs)
#     valid_sampler = SubsetRandomSampler(valid_idxs)

#     train_loader = DataLoader(data,
#         batch_size = batch_size,
#         sampler = train_sampler,
#         num_workers = num_workers,
#         pin_memory = pin_memory
#     )
#     valid_loader = DataLoader(data,
#         batch_size = batch_size,
#         sampler = valid_sampler,
#         num_workers = num_workers,
#         pin_memory = pin_memory
#     )

#     return train_loader, valid_loader

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

def make_loader(data, batch_size = 4, shuffle = True, num_workers = 0, pin_memory = True):
    return DataLoader(data,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = shuffle
    )

# def evaluate(model, criterion, loader, device):
#     model.eval()
#     total_loss = 0
#     num_iters = 0
#     for images, ps in loader:
#         ps = ps.to(device = device)
#         images = images.to(device = device)
#         ps_out = model(images)
#         loss = criterion(ps_out, ps)
#         total_loss += loss.item() # Important to use .item() !
#         num_iters += 1
#     avg_loss = total_loss / num_iters
#     return avg_loss

def evaluate_median(model, criterion, loader, device): # Expects a loader with batch_size = 1
    model.eval()
    x_errors = []
    q_errors = []
    losses = []

    for image, p in loader:
        image = image.to(device = device)
        p_outs = model(image)
        p_out = p_outs[-1].expand(1, -1)
        p_out = p_out.to(device = device)

        p = p.expand(1, -1)
        p = p.to(device = device)

        loss_out = criterion(p_out, p)
        losses.append(loss_out.item())

        x = p[:, :2].cpu().detach().numpy()
        q = p[:, 2:].cpu().detach().numpy()
        x_out = p_out[:, :2].cpu().detach().numpy()
        q_out = p_out[:, 2:].cpu().detach().numpy()
        
        error_x, theta = meters_and_degrees_error(x, q, x_out, q_out)

        x_errors.append(error_x)
        q_errors.append(theta)

    x_error_median = np.median(x_errors)
    q_error_median = np.median(q_errors)
    loss_median = np.median(losses)
    return x_error_median, q_error_median, loss_median

def meters_and_degrees_error(x, q, x_predicted, q_predicted):
    q1 = q / np.linalg.norm(q)
    q2 = q_predicted / np.linalg.norm(q_predicted)
    d = np.abs(np.sum(np.multiply(q1, q2)))
    orientation_error = 2 * np.arccos(d) * 180 / np.pi
    position_error = np.linalg.norm(x - x_predicted)
    return position_error, orientation_error

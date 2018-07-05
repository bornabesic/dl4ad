
import os.path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import torchvision.transforms as transforms

from utils import read_table, lines
from preprocessing import Resize, RandomCrop, SubtractMean, ToTensor, Compose
from transformations import euler_from_quaternion, quaternion_from_euler
from augmentation import augmentations

class DeepLoc(Dataset):

    default_preprocessing = Compose([
        Resize(256), # Rescale so that the smaller edge is 256 pxs
        RandomCrop(size = (224, 224)), # Take a random 224 x 224 crop
        SubtractMean(),
        ToTensor()
    ])

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

    def __init__(self, mode, preprocess = DeepLoc.default_preprocessing):
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

    default_preprocessing = Compose([
        Resize(size = (224, 224)), # Rescale so that the smaller edge is 256 pxs
        SubtractMean(),
        ToTensor()
    ])

    valid_preprocessing = Compose([
        Resize(size = (224, 224)), # Rescale so that the smaller edge is 256 pxs
        SubtractMean(),
        ToTensor()
    ])

    def __init__(self, set_path, mode, preprocess = default_preprocessing, augment = True):
        self.data = []
        self.preprocess = preprocess
        self.augment = augment

        if mode not in ("train", "validation", "test", "visualize"):
            raise ValueError("Invalid mode.")

        self.mode = mode
        
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
                _, _, theta = euler_from_quaternion((qw, qx, qy, qz))
                pose = (x, y, theta)
                image_path = os.path.join(mode_path, filename)
                # image_path = os.path.join(set_path, filename)
                self.data.append((image_path, pose))
        else:
            poses_path = os.path.join(set_path, mode + ".txt")
            for *filenames, x, y, qw, qx, qy, qz in read_table(
                poses_path,
                types = (str, str, str, str, str, str, float, float, float, float, float, float),
                delimiter = " "):
                    _, _, theta = euler_from_quaternion((qw, qx, qy, qz))
                    pose = (x, y, theta)
                    image_paths = map(lambda fn: os.path.join(set_path, fn), filenames)
                    #image_paths = filenames
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
            x, y, theta = pose
        else:

            *image_paths, pose = self.data[idx]
            x, y, theta = pose

            # Randomly switch front three and back three images
            switch_front_and_back = np.random.choice((True, False)) if self.augment else False
#            if switch_front_and_back:
#                image_paths = image_paths[3:] + image_paths[:3]
#                a1, a2, a3 = euler_from_quaternion(q)
#                a3 = a3 + np.sign(a3) * np.pi # rotate the pose by 180°
#                qw, qx, qy, qz = quaternion_from_euler(a1, a2, a3)

            images = map(Image.open, image_paths)


#            # Randomly augment the six images with one effect online
 #           should_augment = np.random.choice((True, False)) if self.augment else False
 #           if should_augment:
 #               x = round(np.random.uniform(len(augmentations))) - 1
 #               images = map(augmentations[x],images)

            if self.preprocess is not None:
                images = map(self.preprocess, images)

            # Concatinate all images (each 3 layer) to one image with 18 layers
            images = tuple(images)
            image = torch.cat(images, dim = 0)
        
        # Hardcoded normalization to [-1,1]. Max. area from dataset is 350m x 350m
   #     print("X vor teiler")
  #      print(x)
        x = x / 350
 #       print("X nach teiler")
#        print(x)
        y = y / 350
        p = torch.Tensor([x, y, theta])

        return (image, p)

from utils import foldr

class PerceptionCarDatasetMerged(Dataset):

    def __init__(self, *dataset_paths, mode, preprocess = PerceptionCarDataset.default_preprocessing, augment = True):
        self.datasets = list(map(lambda dp: PerceptionCarDataset(dp, mode, preprocess, augment), dataset_paths))
        self.size = foldr(lambda i, r: len(i) + r, self.datasets, 0)
        
        self.origin = self.datasets[0].origin
        '''
        offset = origin - reference
        origin = reference + offset
        '''
        for dataset in self.datasets:
            dataset.origin_offset = (dataset.origin - self.origin) / 350 #with hardcoded normalization


    def __getitem__(self, idx):
        for dataset in self.datasets:
            try:
                item = dataset[idx]
                image, pose = item
                x, y, *q = pose
                x, y, _ = torch.Tensor([x, y, 0]) + dataset.origin_offset
                pose = torch.Tensor((x, y, *q))
                return (image, pose)
            except IndexError:
                idx -= len(dataset)

    def __len__(self):
        return self.size
            


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
    # q1 = q / np.linalg.norm(q)
    # q2 = q_predicted / np.linalg.norm(q_predicted)
    # d = np.abs(np.sum(np.multiply(q1, q2)))
    # orientation_error = 2 * np.arccos(d) * 180 / np.pi
    orientation_error = np.abs(q - q_predicted)
    position_error = np.linalg.norm(x - x_predicted)
    return position_error, orientation_error

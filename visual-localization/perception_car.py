from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import os.path
import fnmatch
from random import shuffle
from PIL import Image
from PIL import ImageOps
import PIL
import random
import numpy as np
import transformations as transforms_ros
import torch
import ais_helper


def shuffle_list(*ls):
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)


def searchForFiles(name, path):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, name):
            matches.append(os.path.join(root, filename))
    return matches


def readFiles(path):
    with open(path) as file:
        content = file.readlines()
        content = [x.strip() for x in content]
    return content


def getTQ(vehicle):
    t = (vehicle[0], vehicle[1], vehicle[2])
    q = (vehicle[3], vehicle[4], vehicle[5], vehicle[6])
    return t, q


def calculateCameraMotion(t1, q1, camera_calibration):
    Q_1 = np.array(q1)
    T_1 = transforms_ros.quaternion_matrix(Q_1)
    T_1[0:3, 3] = np.array(t1)

    T_world_camera = torch.from_numpy(T_1).float().view(4, 4)  # Convert to torch tensor

    # Static Base_link to pointgray transform
    T_pg_bl = camera_calibration

    cam_in_bl_1 = T_world_camera.mm(T_pg_bl)

    R_final = cam_in_bl_1[0:3, 0:3].contiguous().view(1, 3, 3)
    T_final = cam_in_bl_1[0:3, 3].contiguous().view(1, 1, 3)
    R_final_inv = cam_in_bl_1[0:3, 0:3].contiguous().view(1, 3, 3)
    T_final_inv = cam_in_bl_1[0:3, 3].contiguous().view(1, 1, 3)

    return R_final, T_final, R_final_inv, T_final_inv

def remapCameraName(path):
    pairs = [
        ("back/center", "zed_back"), # z_back
        ("back/left", "zed_rl"), # z_rear_l
        ("back/right", "zed_rr"), # z_rear_r
        ("front/center", "zed_front"), # z_front
        ("front/left", "zed_fl"), # z_front_l
        ("front/right", "zed_fr") # z_front_r
    ]
    for start, camera_name in pairs:
        if path.startswith(start):
            return camera_name
    
    return None


class PerceptionCarDataset(Dataset):
    def __init__(self, db_path, width, height, calibration_dir, return_paths = False):
        self.width = width
        self.height = height
        self.return_paths = return_paths
        self.calibration_info = {}
        self.db_path = db_path

        self.data = []  # Stores pairs of consecutive image paths

        # Load calibration files
        cal_files = searchForFiles("zed_*.yaml", calibration_dir)
        for cal_i in cal_files:
            print("Read calibration file: %s" % (cal_i))
            filename = os.path.splitext(os.path.basename(cal_i))[0]
            self.calibration_info[filename] = ais_helper.CameraParams(cal_i)

        paths_L_files = searchForFiles("paths.L.txt", db_path)
        paths_R_files = searchForFiles("paths.R.txt", db_path)
        poses_files = searchForFiles("poses.txt", db_path)

        origin_file = searchForFiles("origin.txt", db_path)[0]
        self.origin = torch.Tensor(tuple(map(float, readFiles(origin_file)[0].split("/"))))

        paths_L_files = sorted(paths_L_files)
        paths_R_files = sorted(paths_R_files)
        poses_files = sorted(poses_files)
        files = zip(paths_L_files, paths_R_files, poses_files)

        for paths_L_filename, paths_R_filename, poses_filename in files: # Paths to txt files
            left_images = readFiles(paths_L_filename) # Lines
            right_images = readFiles(paths_R_filename)
            poses = readFiles(poses_filename)

            for line_L, line_R, line_pose in zip(left_images, right_images, poses):
                frame_L1, frame_L2, frame_L3 = line_L.split(" ")  # Current first, then previous
                frame_R1, frame_R2, frame_R3 = line_R.split(" ")  # Current first, then previous
                pose1, pose2, pose3 = filter(None, line_pose.split("/"))

                self.data.append((frame_L1, getTQ(list(filter(None, pose1.split(" "))))))

        self.length = len(self.data)

    def resizeNearest(self, imgs, size):
        for idx in range(len(imgs)):
            imgs[idx] = imgs[idx].resize(size, resample=PIL.Image.NEAREST)
        return imgs

    def resizeLinear(self, imgs, size):
        for idx in range(len(imgs)):
            imgs[idx] = imgs[idx].resize(size, resample=PIL.Image.BILINEAR)
        return imgs

    def resizeLinear1(self, img, size):
        return img.resize(size, resample=PIL.Image.BILINEAR)

    def cropImages(self, imgs, params):
        for idx in range(len(imgs)):
            imgs[idx] = F.crop(imgs[idx], *params)
        return imgs

    def flipImages(self, imgs):
        for idx in range(len(imgs)):
            imgs[idx] = F.hflip(imgs[idx])
        return imgs

    def imagesToTensor(self, imgs):
        for idx in range(len(imgs)):
            imgs[idx] = F.to_tensor(imgs[idx])
        return imgs

    def imageToTensor(self, img):
        return F.to_tensor(img)

    def imagesToTensorNonNorm(self, imgs):
        for idx in range(len(imgs)):
            imgs[idx] = torch.from_numpy(np.expand_dims(np.array(imgs[idx]), 0))
        return imgs

    def normImages(self, imgs):
        for idx in range(len(imgs)):
            imgs[idx] = normalizeForSeg(imgs[idx])
        return imgs

    def __getitem__(self, index):
        image_path, pose = self.data[index]
        t, q = pose

        camera_name = remapCameraName(image_path)

        image_path = os.path.join(self.db_path, image_path)

        # Calculate camera transformation
        R_camera, T_camera, R_final_inv, T_final_inv = calculateCameraMotion(t, q, torch.from_numpy(
                self.calibration_info[camera_name].T_bl_cam).float())

        camera_intrisics = torch.from_numpy(np.copy(self.calibration_info[camera_name].c_m_0))
        camera_intrisics = camera_intrisics.view(1, 3, 3)

        # Read images
        image = Image.open(image_path)

        # Resizing affects the camera focallength.
        w_ratio = float(self.width) / float(image.size[0])
        h_ratio = float(self.height) / float(image.size[1])

        camera_intrisics[0, 0, 0] = camera_intrisics[0, 0, 0] * w_ratio
        camera_intrisics[0, 1, 1] = camera_intrisics[0, 1, 1] * h_ratio
        camera_intrisics[0, 0, 2] = camera_intrisics[0, 0, 2] * w_ratio
        camera_intrisics[0, 1, 2] = camera_intrisics[0, 1, 2] * h_ratio
        new_size = (self.width, self.height)

        image = self.resizeLinear1(image, new_size)
        image = self.imageToTensor(image)

        # NOTE: The calculated camera transformation is from timestep t to t+1
        if self.return_paths:
                return image, R_camera, T_camera, R_final_inv, T_final_inv, self.origin, image_path

        return image, R_camera, T_camera, R_final_inv, T_final_inv, self.origin, camera_intrisics

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


import os.path
from torch.utils.data import Dataset
import cv2

from utils import read_table

class DeepLoc(Dataset):

    def __init__(self, mode):
        self.data = []
        self.size = 0

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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return (image, pose)

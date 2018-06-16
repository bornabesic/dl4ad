import cv2
import numpy as np
import torch
from perception_car import PerceptionCarDataset
from visualize import PosePlotter
from utils import rad2deg
from transformations import euler_from_matrix

def show_image(data, name):
    cv = np.transpose(data.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, cv)

db_path = './loc_dataset/data/'
cal_dir = './loc_dataset/data/calibration'

width = 640
height = 392

images_db = PerceptionCarDataset(
    db_path,
    width, height,
    calibration_dir = cal_dir,
    return_paths = True
)
print("Current number of images:", len(images_db))

images_loader = torch.utils.data.DataLoader(
    images_db,
    batch_size = 1,
    shuffle = False,
    num_workers = 1
)

plotter = PosePlotter(update_interval = 0.5, trajectory = True)

for image, R_camera, T_camera, R_final_inv, T_final_inv, origin, image_path in images_loader:
    _, _, theta = euler_from_matrix(R_camera[0][0])
    azimuth = rad2deg(-theta)
    X = origin + T_camera
    x, y, z = X[0][0][0]
    x = x.item()
    y = y.item()
    lat, lng = PosePlotter.utm2latlng(x, y)
    print(lat, lng, azimuth)
    plotter.update(x, y)
    show_image(image, "Image")
    cv2.waitKey(1)

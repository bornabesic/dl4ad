import os
import cv2
import numpy as np
import torch
from perception_car import PerceptionCarDatasetRaw
from visualize import PosePlotter
from utils import rad2deg
from transformations import euler_from_matrix, quaternion_from_euler

def tensor2cv(tensor, range = 255):
    cv = np.transpose(tensor.cpu().data.numpy().squeeze(), (1, 2, 0))
    cv = cv2.cvtColor(cv, cv2.COLOR_RGB2BGR)
    return cv * range

def show_image(data, name):
    cv2.imshow(name, tensor2cv(data, range = 1))

visualize = False
db_path = './loc_dataset/data/'
cal_dir = './loc_dataset/data/calibration'

images_db = PerceptionCarDatasetRaw(
    db_path,
    calibration_dir = cal_dir,
    return_paths = True
)
print("Current number of images:", len(images_db))

images_loader = torch.utils.data.DataLoader(
    images_db,
    batch_size = 1,
    shuffle = False,
    num_workers = 4
)

plotter = PosePlotter(update_interval = 0.05, trajectory = True)

dataset_path = "PerceptionCarDataset"
os.makedirs(dataset_path, exist_ok = True)

# Write origin to a file
origin_path = os.path.join(dataset_path, "origin.txt")
with open(origin_path, "wt", encoding = "utf-8") as origin_file:
    x, y, z = images_db.origin.numpy()
    print(x, y, z, file = origin_file)

for i, (image, R_camera, T_camera, R_final_inv, T_final_inv, origin, image_path) in enumerate(images_loader):
    a1, a2, theta = euler_from_matrix(R_camera[0][0])
    if visualize:
        azimuth = rad2deg(-theta)
        X = origin + T_camera
        x, y, z = X[0][0][0].numpy()
        lat, lng = PosePlotter.utm2latlng(x, y)
        print(lat, lng, azimuth)
        plotter.update(x, y)
        show_image(image, "Image")
        cv2.waitKey(1)
    else:
        # Save the image as JPEG
        directory, image_name = os.path.split(image_path[0])
        images_directory, _ = os.path.split(directory)
        final_directory, _ = os.path.split(images_directory)
        jpeg_image_name = image_name.replace(".png", ".jpeg")
        dest_path = os.path.join(dataset_path, final_directory)
        os.makedirs(dest_path, exist_ok = True)
        cv2.imwrite(os.path.join(dest_path, jpeg_image_name), tensor2cv(image))

        # Append (image_path, pose) pair to poses.txt
        x_local, y_local, z_local = T_camera[0][0][0].numpy()
        qw, qx, qy, qz = quaternion_from_euler(a1, a2, theta)
        # print(x_local, y_local, qw, qx, qy, qz)
        print (i + 1, "/", len(images_loader))
        poses_path = os.path.join(dest_path, "poses.txt")
        with open(poses_path, "a", encoding = "utf-8") as poses_file:
            print(jpeg_image_name, x_local, y_local, qw, qx, qy, qz, file = poses_file)

print("Done!")

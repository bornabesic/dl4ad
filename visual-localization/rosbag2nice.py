
import os
import torch
from perception_car import searchForFiles, getTQ, readFiles, calculateCameraMotion
from preprocessing import ToRGB
from PIL import Image
from transformations import euler_from_quaternion, euler_from_matrix, quaternion_from_euler
from utils import rad2deg

import ais_helper


# TODO Supply through CLI arguments
rosbag_path = "/media/borna/Backup/visual_loc_group/"
dest_path = "/mnt/data/PerceptionCarDataset/"

os.makedirs(dest_path, exist_ok = True)

paths_L_files = sorted(searchForFiles("z_*_drive_*_left.txt", rosbag_path))
poses_files = sorted(searchForFiles("z_*_drive_*_vehicle.txt", rosbag_path))

files = zip(paths_L_files, poses_files)

# Load calibration files
calibration_info = {}
cal_files = searchForFiles("zed_*.yaml", rosbag_path)
for cal_i in cal_files:
    print("Read calibration file: {}".format(cal_i))
    filename = os.path.splitext(os.path.basename(cal_i))[0]
    calibration_info[filename] = ais_helper.CameraParams(cal_i)

data = []

for paths_L_filename, poses_filename in files: # Paths to txt files
    # Lines
    left_images = readFiles(paths_L_filename)
    poses = readFiles(poses_filename)

    for line_L, line_pose in zip(left_images, poses):
        frame_L1, frame_L2, frame_L3 = line_L.split(" ")  # Current first, then previous
        pose1, pose2, pose3 = filter(None, line_pose.split("/"))

        pose_list = list(filter(None, pose1.split(" ")))

        data.append((frame_L1, pose_list))

origin_file = searchForFiles("z_front_drive_*_origin.txt", rosbag_path)[0]
origin = tuple(map(float, readFiles(origin_file)[0].split("/")))

origin_path = os.path.join(dest_path, "origin.txt")
with open(origin_path, "w", encoding = "utf-8") as origin_file:
    print(*origin, sep = " ", file = origin_file)

dir_map = {
    "front": os.path.join("front", "center"),
    "front_l": os.path.join("front", "left"),
    "front_r": os.path.join("front", "right"),
    "back": os.path.join("back", "center"),
    "rear_l": os.path.join("back", "left"),
    "rear_r": os.path.join("back", "right"),
}

rgb_converter = ToRGB()

# Camera to calibration file
camera_calibration_map = {
    "left_z_back": "zed_back", # z_back
    "left_z_rear_l": "zed_rl", # z_rear_l
    "left_z_rear_r": "zed_rr", # z_rear_r
    "left_z_front": "zed_front", # z_front
    "left_z_front_l": "zed_fl", # z_front_l
    "left_z_front_r": "zed_fr" # z_front_r
}

for i, (image_path, pose) in enumerate(data):
    print("{:5.2f} %".format((i + 1) / len(data) * 100), end = "\r")
    image_path_normalized = os.path.normpath(image_path)
    rest1, image_name = os.path.split(image_path_normalized)
    rest2, camera_dir = os.path.split(rest1)
    rest3, drive_dir = os.path.split(rest2)
    abs_image_path = os.path.join(rosbag_path, drive_dir, camera_dir, image_name) # Reading
    image = Image.open(abs_image_path)
    image_rgb = rgb_converter(image)

    # Remap camera name
    camera_name = camera_calibration_map[camera_dir]

    x, y, z, *q = pose
    t = (x, y, z)

    # Calculate camera transformation
    R_camera, T_camera, R_final_inv, T_final_inv = calculateCameraMotion(t, q, torch.from_numpy(
            calibration_info[camera_name].T_bl_cam).float())

    a1, a2, theta = euler_from_matrix(R_camera[0])
    qw, qx, qy, qz = quaternion_from_euler(a1, a2, theta)
    pose = x, y, qw, qx, qy, qz

    # Saving
    image_name, ext = image_name.split(".")
    jpeg_image_name = ".".join((image_name, "jpeg"))

    stereo_ch, camera_id = camera_dir.split("_z_")
    jpeg_image_dir = os.path.join(dest_path, dir_map[camera_id])

    jpeg_image_path = os.path.join(jpeg_image_dir, jpeg_image_name)
    os.makedirs(jpeg_image_dir, exist_ok = True)
    image_rgb.save(jpeg_image_path, "JPEG")

    poses_path = os.path.join(jpeg_image_dir, "poses.txt")
    with open(poses_path, "a", encoding = "utf-8") as poses_file:
        print(os.path.join(dir_map[camera_id], jpeg_image_name), *pose, file = poses_file, sep = " ")

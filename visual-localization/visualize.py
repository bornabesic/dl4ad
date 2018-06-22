#!/usr/bin/env python

import torch
import cv2
import numpy as np
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import matplotlib.pyplot as plt
import matplotlib
import utm
from dataset import PerceptionCarDataset
from transformations import euler_from_quaternion
from utils import rad2deg

class PosePlotter:

    UTM_zone = (32, "U")
    lat_max, lng_max = 48.01507, 7.8364
    lat_min, lng_min = 48.01268, 7.83086

    def __init__(self, update_interval = 1, trajectory = False):
        self.update_interval = update_interval
        self.trajectory = trajectory
        imagery = OSM()

        self.fig = plt.figure()
        ax = self.fig.add_subplot(1, 1, 1, projection = imagery.crs)
        
        ax.set_extent([
            PosePlotter.lng_min,
            PosePlotter.lng_max,
            PosePlotter.lat_min,
            PosePlotter.lat_max
        ], ccrs.PlateCarree())

        ax.add_image(imagery, 17)

        self.data = plt.plot(
            [0], [0],
            color = "red",
            marker = PosePlotter.get_marker(),
            transform = ccrs.Geodetic(),
        )[0]

        self.marker_reference = self.data.get_marker()

        if self.trajectory:
            self.lats = []
            self.lngs = []

        plt.ion()
        plt.draw()
        plt.show()
        
    def update(self, x, y, theta):
        lat, lng = PosePlotter.utm2latlng(x, y)
        self.data.set_visible(True)

        new_marker = self.marker_reference.transformed(matplotlib.transforms.Affine2D().rotate_deg(rad2deg(theta)))
        self.data.set_marker(new_marker)
        
        if self.trajectory:
            self.lats.append(lat)
            self.lngs.append(lng)
            i = len(self.lats)
            self.data.set_xdata([self.lngs[: (i + 1)]])
            self.data.set_ydata([self.lats[: (i + 1)]])
        else:
            self.data.set_xdata([lng])
            self.data.set_ydata([lat])

        plt.draw()
        plt.pause(self.update_interval)

    @staticmethod
    def utm2latlng(x, y):
        return utm.to_latlon(x, y, *PosePlotter.UTM_zone)

    @staticmethod
    def get_marker():
        verts = [
            (-1, -1),
            (0, 1),
            (1, -1),
            (0, 0),
            (-1, -1)
        ]

        codes = [
            matplotlib.path.Path.MOVETO,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.LINETO,
            matplotlib.path.Path.CLOSEPOLY
        ]

        return matplotlib.path.Path(verts, codes)


if __name__ == "__main__":

    plotter = PosePlotter(update_interval = 0.02, trajectory = False)

    data = PerceptionCarDataset("visualize", preprocess = None)

    for image, pose in data:
        x, y, qw, qx, qy, qz = pose
        # Global pose
        x, y, _ = torch.Tensor([x, y, 0]) + data.origin
        lat, lng = PosePlotter.utm2latlng(x, y)
        _, _, theta = euler_from_quaternion([qw, qx, qy, qz])
        azimuth = rad2deg(-theta)
        print(lat, lng, azimuth)
        plotter.update(x, y, theta)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("front/center", image_cv)
        cv2.waitKey(1)

#!/usr/bin/env python

import torch
import cv2
import numpy as np
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import matplotlib.pyplot as plt
import matplotlib
import utm
from dataset import PerceptionCarDataset, PerceptionCarDatasetMerged
from transformations import euler_from_quaternion
from utils import rad2deg

class PosePlotter:

    UTM_zone = (32, "U")
    lat_max, lng_max = 48.01460, 7.8364
    lat_min, lng_min = 48.01230, 7.83086

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

        self.poses = dict()
        self.colors = PosePlotter.color_generator()

        plt.ion()
        plt.hold(True)
        plt.draw()
        plt.show()

    def register(self, id, color):
        self.poses[id] = dict(
            data = plt.plot(
                [0], [0],
                color = color,
                marker = "." if self.trajectory else PosePlotter.get_marker(),
                transform = ccrs.Geodetic(),
                label = id,
                alpha = 0.5
            )[0],
            # label = plt.text(0, 0,
            #     id,
            #     transform = ccrs.Geodetic()
            # ),
            lats = [],
            lngs = []
        )
        self.poses[id]["marker"] = self.poses[id]["data"].get_marker()

        
    def update(self, id, x, y, theta):
        lat, lng = PosePlotter.utm2latlng(x, y)
        
        pose = self.poses[id]
        lats, lngs = pose["lats"], pose["lngs"]
        data = pose["data"]
        # label = pose["label"]
        data.set_visible(True)

        
        
        if self.trajectory:
            lats.append(lat)
            lngs.append(lng)
            i = len(lats)
            data.set_xdata([lngs[: (i + 1)]])
            data.set_ydata([lats[: (i + 1)]])
        else:
            marker_reference = pose["marker"]
            new_marker = marker_reference.transformed(matplotlib.transforms.Affine2D().rotate_deg(rad2deg(theta)))
            data.set_marker(new_marker)
            data.set_xdata([lng])
            data.set_ydata([lat])
            # label.set_x(lng)
            # label.set_y(lat)

    def draw(self):
        plt.legend()
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

    @staticmethod
    def color_generator():
        # colors = list(matplotlib.colors.BASE_COLORS.keys())
        colors = ["red", "blue"]
        idx = 0
        while True:
            yield colors[idx]
            idx += 1
            if idx >= len(colors):
                idx = 0

if __name__ == "__main__":

    plotter = PosePlotter(update_interval = 0.02, trajectory = False)
    plotter.register("GT", "red")
    # plotter.register("x", "blue") # TEST

    data = PerceptionCarDatasetMerged(
        "PerceptionCarDataset",
        "PerceptionCarDataset2",
        mode = "visualize",
        preprocess = None,
        augment = False
    )

    for image, pose in data:
        x, y, theta = PerceptionCarDataset.unnormalize(*pose)
        lat, lng = PosePlotter.utm2latlng(x, y)
        azimuth = rad2deg(-theta.item())
        print(lat, lng, azimuth)
        plotter.update("GT", x, y, theta)
        plotter.draw()
        # plotter.update("x", x + 10, y + 10, theta + 3.14) # TEST
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow("front/center", image_cv)
        cv2.waitKey(1)

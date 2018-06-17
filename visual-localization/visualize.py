import torch
import cv2
import numpy as np
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import matplotlib.pyplot as plt
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
            color="red",
            marker='.',
            transform=ccrs.Geodetic(),
        )[0]

        if self.trajectory:
            self.lats = []
            self.lngs = []

        plt.ion()
        plt.draw()
        plt.show()
        
    def update(self, x, y):
        lat, lng = PosePlotter.utm2latlng(x, y)
        self.data.set_visible(True)
        
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


if __name__ == "__main__":

    camera = "front/center"
    plotter = PosePlotter(update_interval = 0.02, trajectory = False)
    data = PerceptionCarDataset("all", preprocess = None)

    for image, camera_id, pose in filter(lambda item: item[1] == camera, data):
        x, y, qw, qx, qy, qz = pose
        # Global pose
        x, y, _ = torch.Tensor([x, y, 0]) + data.origin
        lat, lng = PosePlotter.utm2latlng(x, y)
        _, _, theta = euler_from_quaternion([qw, qx, qy, qz])
        azimuth = rad2deg(-theta)
        print(lat, lng, azimuth)
        plotter.update(x, y)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(camera, image_cv)
        cv2.waitKey(1)

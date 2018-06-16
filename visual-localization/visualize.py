import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import matplotlib.pyplot as plt
import utm

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
    x, y = 412940.751955, 5318560.37949
    rot = 0

    plotter = PosePlotter(update_interval = 10, trajectory = False)
    plotter.update(x, y)

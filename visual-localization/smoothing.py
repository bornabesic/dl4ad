import numpy as np
from scipy.stats import multivariate_normal

class TrajectorySmoother:

    def __init__(self, num_points = 10):
        self.positions = []

        self.position_diffs = []
        self.num_points = num_points
        # self.mean = 0.5
        # self.variance = 0.5

    def update(self, x, y, theta):
        if len(self.positions) >= self.num_points:
            self.positions.pop(0)
        self.positions.append(np.array((x, y)))

        self.position_diffs.clear()

        for i in range(1, len(self.positions)):
            xy1 = self.positions[i - 1]
            xy2 = self.positions[i]
            self.position_diffs.append(xy2 - xy1)

        self.translation_mean = np.mean(self.position_diffs, axis = 0)
        self.translation_cov = np.cov(self.position_diffs, rowvar = False)

    def probability(self, x, y, theta):
        diff = np.array((x, y)) - self.positions[-1]
        if np.isnan(self.translation_mean).any():
            return 1

        try:
            return multivariate_normal.pdf(diff, mean = self.translation_mean, cov = self.translation_cov)
        except np.linalg.linalg.LinAlgError:
            return 1

    def smooth(self, x, y, theta):
        p_xy = self.probability(x, y, theta)
        return p_xy * np.array((x, y)) + (1 - p_xy) * (np.median(self.positions, axis = 0))
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cosine

class TrajectorySmoother:

    def __init__(self, num_points = 15):
        self.positions = []

        self.position_diffs = []
        self.num_points = num_points

        self.last_position = 0

    def update(self, x, y, theta):
        if len(self.positions) >= self.num_points:
            self.positions.pop(0)
        
        self.positions.append(np.array((x, y)))

        if len(self.positions) >= 2:
            xy2 = self.positions[-1]
            xy1 = self.positions[-2]
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

        tx, ty = np.array((x, y)) - self.last_position
        theta_movement = np.arctan2(ty, tx)

        theta = -theta - np.pi / 2 
        theta_diff = np.abs(theta - theta_movement)
        theta_error = min(np.pi - theta_diff, theta_diff)
        p_theta = max(1 - theta_error / np.pi, 0)

        direction = np.array((np.cos(theta), -np.sin(theta)))
        speed = np.median(list(map(np.linalg.norm, self.position_diffs[:-5]))) if len(self.position_diffs) > 5 else 1
        

        p = p_xy * p_theta if self.last_position is not 0 else p_xy

        if self.last_position is None:
            self.last_position = np.mean(self.positions, axis = 0)
        self.last_position = p * np.array((x, y)) + (1 - p) * 0.9 * (self.last_position + 1.75 * direction) + (1 - p) * 0.1 * np.mean(self.positions, axis = 0)
        return self.last_position

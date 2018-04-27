
import torch
import numpy as np
from skimage import transform

class Resize(object):

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, image):
        return transform.resize(image, output_shape = (self.w, self.h))

class SubtractMean(object):

    def __call__(self, image):
        ch1 = image[:, :, 0]
        ch2 = image[:, :, 1]
        ch3 = image[:, :, 2]

        ch1_new = ch1 - np.mean(ch1)
        ch2_new = ch2 - np.mean(ch2)
        ch3_new = ch3 - np.mean(ch3)

        return np.stack((ch1_new, ch2_new, ch3_new), axis = 0)

class MakeTensor(object):

    def __call__(self, obj):
        return torch.from_numpy(obj)


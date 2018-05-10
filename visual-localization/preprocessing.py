
import cv2

# TODO Rescale the image so the smallest dimension is 256 pixels
class Rescale:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        return image

# TODO Extract random 224 x 224 crops
class RandomCrops:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image):
        return image

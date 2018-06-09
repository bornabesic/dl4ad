
from torchvision.transforms import CenterCrop, FiveCrop, RandomCrop, Resize, Compose, ToTensor
import numpy as np

# Convert the image from RGBA to RGB
class ToRGB:

    def __call__(self, image):
        return image.convert("RGB")

# Subtract the mean from every channel in the image
class SubtractMean(object):

    def __call__(self, image):
        image_array = np.array(image, dtype = np.float32)

        ch1 = image_array[:, :, 0]
        ch2 = image_array[:, :, 1]
        ch3 = image_array[:, :, 2]

        ch1_new = ch1 - np.mean(ch1)
        ch2_new = ch2 - np.mean(ch2)
        ch3_new = ch3 - np.mean(ch3)

        subtracted_mean_image = np.stack((ch1_new, ch2_new, ch3_new), axis = 2)
        return subtracted_mean_image

default_preprocessing = Compose([
    Resize(256), # Rescale so that the smaller edge is 256 pxs
    RandomCrop(size = (224, 224)), # Take a random 224 x 224 crop
    SubtractMean(),
    ToTensor()
])

validation_resize = Resize(256)
validation_crops = FiveCrop(size = (224, 224))
validation_tensor = Compose([
    SubtractMean(),
    ToTensor()
])

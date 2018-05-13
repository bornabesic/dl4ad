#!/usr/bin/env python3

# TODO Change brightness
class ChangeBrightness:

    def __init__(self, value):
        self.value = value

    def __call__(self, image):
        return image

# TODO Change contrast
class ChangeContrast:

    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, image):
        return image

# TODO Add Gaussian blur
class GaussianBlur:

    def __init__(self, kernel_width, kernel_height, mean, std_x, std_y):
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.mean = mean
        self.std_x = std_x
        self.std_y = std_y

    def __call__(self, image):
        return image

# TODO Add Gaussian noise
class GaussianNoise:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return image

# TODO Add salt-and-pepper noise
class SaltAndPepperNoise:

    def __init__(self, percentage):
        self.percentage = percentage

    def __call__(self, image):
        return image

# TODO Region dropout
# (mask out random rectangles, each taking ~1% of the image)
class RegionDropout:

    def __init__(self, percentage, number):
        self.percentage = percentage
        self.number = number

    def __call__(self, image):
        return image

if __name__ == "__main__":
    from dataset import DeepLoc

    # Load the dataset
    train_data = DeepLoc("train")
    test_data = DeepLoc("test")

    # Define the augmentations
    augmentations = [
        # ChangeBrightness(),
        # ChangeContrast(),
        # GaussianBlur(),
        # GaussianNoise(),
        # SaltAndPepperNoise(),
        # RegionDropout()
    ]

    for augment in augmentations:

        # Iterate the train data
        for image, pose in train_data:
            # Augment the image
            image_augmented = augment(image)
            # TODO Save the image (depends on the lib)

        # Iterate the train data
        for image, pose in test_data:
            # Augment the image
            image_augmented = augment(image)

            # TODO Save the image (depends on the lib)
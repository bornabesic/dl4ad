#!/usr/bin/env python3
import PIL
from PIL import ImageEnhance
from PIL import Image
from PIL import ImageFilter
import numpy as np
import os

# Identity transformation
# (Returns the original image)
class Identity:

    def __call__(self, image):
        return image

# Change brightness
class ChangeBrightness:
    # A value between 0.1 and 4 makes sense
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.multiplier)
        return image

# Change contrast
class ChangeContrast:

    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, image):
        # Change Contrast: value between 0.1 and 5 or 6 makes sense
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.multiplier)
        return image

# Add Gaussian blur
class GaussianBlur:

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        image = image.filter(ImageFilter.GaussianBlur(self.radius))
        return image

# Add Gaussian noise
class GaussianNoise:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image_array = np.asarray(image, dtype=np.float64)
        image_array.setflags(write=1)
        height, width, channels = image_array.shape
        noise = np.random.normal(loc = self.mean, scale = self.std, size = (height, width, channels - 1))
        image_array[:, :, 0:3] += noise
        image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
        return image

# TODO Add salt-and-pepper noise
class SaltAndPepperNoise:

    def __init__(self, percentage):
        self.percentage = percentage

    def __call__(self, image):
        # Convert image to array for changing a single pixel
        image_array = np.asarray(image)
        image_array.setflags(write=1)
        
        # Get information about image-size
        columns = np.shape(image_array)[0]
        rows = np.shape(image_array)[1]
        num_pixeltochange = columns * rows * self.percentage / 2

        # Set randomly chosen single pixels to white in a loop
        counter = num_pixeltochange
        while counter > 1:
            x = np.random.randint(0,columns)
            y = np.random.randint(0,rows)
            image_array[x,y,:] = (0, 0, 0, 255)
            counter = counter -1

        # Set randomly chosen single pixels to black in a loop
        counter = num_pixeltochange
        while counter > 1:
            x = np.random.randint(0,columns)
            y = np.random.randint(0,rows)
            image_array[x,y,:] = 0
            counter = counter -1

        # Form an image from an array 
        image = Image.fromarray(np.uint8(image_array))
        return image

# TODO Region dropout
# (mask out random rectangles, each taking ~1% of the image)
class RegionDropout:

    def __init__(self, percentage, number):
        self.percentage = percentage
        self.number = number

    def __call__(self, image):
        # Convert image to array and get its size
        image_array = np.asarray(image)
        image_array.setflags(write=1)
        columns = np.shape(image_array)[0]
        rows = np.shape(image_array)[1]

        # Set values for Dropout rectangles
        boxwidth = np.round(columns * np.sqrt(self.percentage))
        boxhight = np.round(rows * np.sqrt(self.percentage))

        for i in list(range(self.number)):
            x = np.random.randint(0,columns)
            y = np.random.randint(0,rows)
            y_start = y
            x_end = x + boxwidth
            y_end = y + boxhight
            while x < columns and x < x_end:
                while y < rows and y < y_end:
                    image_array[x,y,:] = (0, 0, 0, 255)
                    y = y + 1
                x = x + 1
                y = y_start

        image = Image.fromarray(np.uint8(image_array))
        return image

if __name__ == "__main__":
    from dataset import DeepLoc

    # Load the dataset
    train_data = DeepLoc("train", preprocess = None)
    test_data = DeepLoc("test", preprocess = None)

    # Make directory structure
    train_path = os.path.join("DeepLocAugmented", "train")
    test_path = os.path.join("DeepLocAugmented", "test")
    poses_train = os.path.join(train_path, "poses.txt")
    poses_test = os.path.join(train_path, "poses.txt")
    os.makedirs(train_path, exist_ok = True)
    os.makedirs(test_path, exist_ok = True)

    # Define the augmentations
    augmentations = [
        Identity(),
        ChangeBrightness(0.5),
        ChangeBrightness(2),
        ChangeContrast(0.5),
        ChangeContrast(2),
        GaussianBlur(2),
        GaussianBlur(3),
        GaussianNoise(0, 10),
        SaltAndPepperNoise(0.1),
        RegionDropout(0.01, 10),
    ]

    # Iterate the train data
    i = 0
    for image, pose in train_data:
        x, y, z, qw, qx, qy, qz = pose

        for augment in augmentations:
            image_name = "Image{}.png".format(i)
            image_path = os.path.join(train_path, image_name)

            # Augment the image
            image_augmented = augment(image)
            
            # Save the image
            image_augmented.save(image_path)
            i += 1

            # Write to poses.txt
            with open(poses_train, "a", encoding = "utf-8") as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(image_path, x, y, z, qw, qx, qy, qz), file = f)

    # Iterate the test data
    i = 0
    for image, pose in test_data:
        x, y, z, qw, qx, qy, qz = pose

        for augment in augmentations:
            image_name = "Image{}.png".format(i)
            image_path = os.path.join(test_path, image_name)

            # Augment the image
            image_augmented = augment(image)

            # Save the image
            image_augmented.save(image_path)
            i += 1

            # Write to poses.txt
            with open(poses_test, "a", encoding = "utf-8") as f:
                print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(image_path, x, y, z, qw, qx, qy, qz), file = f)

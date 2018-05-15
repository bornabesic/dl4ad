#!/usr/bin/env python3
import PIL
from PIL import ImageEnhance
from PIL import Image
from PIL import ImageFilter
import numpy as np

# Change brightness
class ChangeBrightness:
    # A value between 0.1 and 4 makes sense
    def __init__(self, value):
        self.value = value

    def __call__(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.value)
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

    # def __init__(self, kernel_width, kernel_height, mean, std_x, std_y):
    #     self.kernel_width = kernel_width
    #     self.kernel_height = kernel_height
    #     self.mean = mean
    #     self.std_x = std_x
    #     self.std_y = std_y

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
        image_array = np.asarray(image)
        noise = self.std * np.random.randn(np.shape(image_array)[0],np.shape(image_array)[1],np.shape(image_array)[2]) + self.mean
        image_array = image_array + noise
        image = Image.fromarray(np.uint8(image_array))
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
            image_array[x,y,:] = 255
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
                    image_array[x,y,:] = 0
                    y = y + 1
                x = x + 1
                y = y_start

        image = Image.fromarray(np.uint8(image_array))
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
            # image_augmented.save("savetest.jpg")

        # Iterate the test data
        for image, pose in test_data:
            # Augment the image
            image_augmented = augment(image)

            # TODO Save the image (depends on the lib)

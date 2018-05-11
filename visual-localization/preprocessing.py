
from torchvision.transforms import RandomCrop, Resize

# Rescale the image so that the smaller edge of the image
# is matched to 'smaller_size'.
class RescaleSmaller:

    def __init__(self, smaller_size):
        self.transform = Resize(smaller_size)

    def __call__(self, image):
        return self.transform(image)

# Extract ’number’ random crops of size ’width’ x ’height’.
class RandomCrops:

    def __init__(self, width, height, number):
        self.number = number
        self.transform = RandomCrop(size = (height, width))

    def __call__(self, image):
        crops = []
        for _ in range(self.number):
            crops.append(self.transform(image))
        return crops

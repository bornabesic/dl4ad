
from torchvision.transforms import RandomCrop, Resize, Compose, ToTensor

default_preprocessing = Compose([
    Resize(256), # Rescale so that the smaller edge is 256 pxs
    RandomCrop(size = (224, 224)), # Take a random 224 x 224 crop
    ToTensor()
])

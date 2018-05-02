from torchvision import transforms
from preprocessing import Resize, SubtractMean, MakeTensor

# Transformations
shared_transform = transforms.Compose([
    Resize(32, 32),
    SubtractMean(),
    MakeTensor()
])

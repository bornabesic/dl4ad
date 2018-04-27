
import os
from os.path import join
from torch.utils.data import Dataset
from utils import lines, one_hot_encoding
from skimage import io

class GTSRB(Dataset):
    def __init__(self, training_path, transform = None):
        self.transform = transform
        self.class_dirs = [o for o in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, o))]

        self.num_classes = len(self.class_dirs)
        
        self.data = []
        for d in self.class_dirs:
            csv_file_name = "GT-{}.csv".format(d)
            annotation_path = os.path.join(training_path, d, csv_file_name)
            self.data += GTSRB.read_annotation(annotation_path)
        
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path, width, height, klass = self.data[idx]
        sample = io.imread(image_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return (sample, one_hot_encoding(klass, self.num_classes))

    @staticmethod
    def read_annotation(annotation_path):
        data = []
        for line in lines(annotation_path):
            cells = line.split(";")
            if cells[0] == "Filename":
                continue
            image_path, width, height, klass = os.path.join(os.path.dirname(annotation_path), cells[0]), int(cells[1]), int(cells[2]), int(cells[-1])
            data.append((image_path, width, height, klass))
        return data


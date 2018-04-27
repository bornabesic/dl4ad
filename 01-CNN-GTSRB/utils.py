
import numpy as np
import torch

def lines(file_path, encoding = "utf-8"):
    with open(file_path, "rt", encoding = encoding) as f:
        for line in f:
            yield line.rstrip("\n")

def one_hot_encoding(which, size):
    enc = np.zeros(size)
    enc[which] = 1
    return torch.from_numpy(enc)
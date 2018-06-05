import time
import torch

def lines(file_path, encoding = "utf-8"):
    with open(file_path, "rt", encoding = encoding) as f:
        for line in f:
            yield line.rstrip("\n")

def read_table(file_path, types, delimiter = ","):
    expected_cols = len(types)
    for line in lines(file_path):
        cols = line.split(delimiter)
        try:
            true_cols = [t(c) for (t, c) in zip(types, cols)]
            if len(true_cols) == expected_cols:
                yield tuple(true_cols)
        except ValueError:
            pass

def print_torch_cuda_mem_usage():
    alloc_mem = torch.cuda.memory_allocated() / 1024 / 1024
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    alloc_percentage = (alloc_mem / max_mem) * 100
    print("CUDA memory usage: {:0.2f} MiB / {:1.2f} MiB ({:2.2f} %)".format(alloc_mem, max_mem, alloc_percentage))


class Stopwatch:

    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        return (self.end_time - self.start_time) / 60


class Logger:

    def __init__(self, filename, print_to_stdout = False):
        self.print_to_stdout = print_to_stdout
        self.file = open(filename, "w", encoding = "utf-8")

    def log(self, message = "", end = "\n"):
        if self.print_to_stdout:
            print(message, end = end)
        print(message, end = end, file = self.file)

    def close(self):
        self.file.close()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.close()

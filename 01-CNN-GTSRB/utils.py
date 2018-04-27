
def lines(file_path, encoding = "utf-8"):
    with open(file_path, "rt", encoding = encoding) as f:
        for line in f:
            yield line.rstrip("\n")
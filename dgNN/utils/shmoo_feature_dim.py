import os

import pickle, sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))


def mean(arr):
    return sum(arr) / len(arr)


def shmoo_batch_size():
    datasets = ["Peptides-func", "CLUSTER", "CIFAR10", "PATTERN"]

    format = "hyper"
    batch_sizes = [2**i for i in range(8, 13)]
    dim = 64

    for dataset in datasets:
        result_dir = os.path.join(
            "/workspace2/fuse_attention", "dataset", dataset, "gat"
        )
        print(dataset)
        for bs in batch_sizes:
            output = os.path.join(result_dir, f"{format}_dim{dim}_bs{bs}_result.pkl")
            with open(output, "rb") as f:
                time_no_fuse, time_fuse = pickle.load(f)

            print(mean(time_fuse))


def shmoo_feature_dim():
    # datasets = ["PATTERN", "MNIST", "CIFAR10", "COCO-SP", "PascalVOC-SP"]
    datasets = ["PATTERN", "MNIST", "CIFAR10", "PascalVOC-SP"]

    format = "csr"
    dims = [2**i for i in range(5, 10)]
    bs = "1024"

    for dataset in datasets:
        result_dir = os.path.join(
            "/workspace2/fuse_attention", "dataset", dataset, "gt"
        )
        print(dataset)
        for dim in dims:
            output = os.path.join(result_dir, f"{format}_dim{dim}_bs{bs}_result.pkl")
            with open(output, "rb") as f:
                time_no_fuse, time_fuse = pickle.load(f)

            print(mean(time_fuse))


if __name__ == "__main__":
    shmoo_batch_size()

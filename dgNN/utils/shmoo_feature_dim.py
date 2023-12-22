import os

import pickle, sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))


def mean(arr):
    return sum(arr) / len(arr)


def shmoo_batch_size():
    datasets = ["PATTERN"]

    formats = ["softmax", "csr", "hyper"]
    batch_sizes = [2**i for i in range(6, 12)]
    dim = 64

    for dataset in datasets:
        print(dataset)
        for bs in batch_sizes:
            print(bs)
            for i, format in enumerate(formats):
                result_dir = os.path.join(
                    "/workspace2/fuse_attention", "dataset", dataset, "gt"
                )
                output = os.path.join(
                    result_dir, f"{format}_dim{dim}_bs{bs}_result.pkl"
                )
                with open(output, "rb") as f:
                    time_no_fuse, time_fuse = pickle.load(f)
                if i == 0:
                    print(mean(time_no_fuse))
                print(mean(time_fuse))


def shmoo_feature_dim():
    datasets = ["PATTERN"]
    formats = ["softmax", "csr", "hyper"]
    # dims = [2**i for i in range(4, 9)]
    dims = [64]

    bs = "1024"

    for dataset in datasets:
        result_dir = os.path.join(
            "/workspace2/fuse_attention", "dataset", dataset, "gt"
        )
        print(dataset)
        for dim in dims:
            print("dim", dim)
            for i, format in enumerate(formats):
                output = os.path.join(
                    result_dir, f"{format}_dim{dim}_bs{bs}_result.pkl"
                )
                with open(output, "rb") as f:
                    time_no_fuse, time_fuse = pickle.load(f)
                if i == 0:
                    print(mean(time_no_fuse))
                print(mean(time_fuse))


if __name__ == "__main__":
    shmoo_batch_size()
    shmoo_feature_dim()

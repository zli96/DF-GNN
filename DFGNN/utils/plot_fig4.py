import os

import pickle, sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))


def mean(arr):
    return sum(arr) / len(arr)


def shmoo_feature_dim():
    conv = "gt"
    dataset = "PATTERN"
    bs = "1024"
    formats = ["softmax", "csr", "hyper"]
    dims = [2**i for i in range(4, 9)]

    result_dir = os.path.join("/workspace2/fuse_attention", "dataset", dataset, conv)
    print(dataset)
    for dim in dims:
        print("dim", dim)
        for i, format in enumerate(formats):
            output = os.path.join(result_dir, f"{format}_dim{dim}_bs{bs}_result.pkl")
            with open(output, "rb") as f:
                time_no_fuse, time_fuse = pickle.load(f)
            if i == 0:
                print(mean(time_no_fuse))
            print(mean(time_fuse))


if __name__ == "__main__":
    shmoo_feature_dim()

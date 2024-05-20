import os

import pickle, sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))


def mean(arr):
    return sum(arr) / len(arr)


def plot_full_graph_super_node():
    dim = 128
    datasets = ["ppa", "reddit", "protein"]
    formats = ["csr_gm", "cugraph", "softmax_gm", "tiling"]
    convs = ["gt", "agnn", "gat"]

    for conv in convs:
        print(conv)
        if conv == "gat":
            formats = ["csr", "cugraph", "softmax_gm", "tiling"]
        for dataset in datasets:
            result_dir = os.path.join(os.getcwd(), "dataset", dataset, conv)
            print(dataset)
            for i, format in enumerate(formats):
                output = os.path.join(result_dir, f"{format}_dim{dim}_result.pkl")
                with open(output, "rb") as f:
                    time_no_fuse, time_fuse = pickle.load(f)
                if i == 0:
                    print(mean(time_no_fuse))
                print(mean(time_fuse))


if __name__ == "__main__":
    plot_full_graph_super_node()

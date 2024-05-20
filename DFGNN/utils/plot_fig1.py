import os

import pickle, sys


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))


def mean(arr):
    return sum(arr) / len(arr)


def plot_batch_graph():
    convs = ["gt", "agnn", "gat"]
    datasets = [
        "COCO-SP",
        "PascalVOC-SP",
        "MNIST",
        "CIFAR10",
        "CLUSTER",
        "PATTERN",
    ]

    formats = ["pyg", "csr", "cugraph", "softmax", "hyper"]
    print("formats", formats)
    dim = 128
    bs = "1024"
    for conv in convs:
        print(conv)
        for dataset in datasets:
            result_dir = os.path.join(os.getcwd(), "dataset", dataset, conv)
            print(dataset)
            times = []
            for i, format in enumerate(formats):
                output = os.path.join(
                    result_dir, f"{format}_dim{dim}_bs{bs}_result.pkl"
                )
                if os.path.exists(output):
                    with open(output, "rb") as f:
                        time_no_fuse, time_fuse = pickle.load(f)
                    if i == 1 and times[0] == "OOM":
                        times = [mean(time_no_fuse)] + times
                    elif i == 0:
                        times.append(mean(time_no_fuse))
                    times.append(mean(time_fuse))
                else:
                    times.append("OOM")
            for time in times:
                print(time)


if __name__ == "__main__":
    plot_batch_graph()

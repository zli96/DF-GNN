import os

import pickle, sys

import dgl.sparse as dglsp
import matplotlib.pyplot as plt
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))


import argparse

from dgl.dataloading import GraphDataLoader

from tqdm import tqdm
from util import load_data_full_graph, load_dataset_fn, mkdir, parser_argument


def mean(arr):
    return sum(arr) / len(arr)


def figure_boxplot(dataset_name, fig_name, data, mean, std):
    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(data)
    plt.title(
        f"{dataset_name} {fig_name}, mean {mean:.2f}, std {std:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_{fig_name}.png")


def batch_graph_statistics(args):
    dataset, train_fn = load_dataset_fn(args.dataset, "/workspace2/dataset")
    train_dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    num_nodes = []  # 每个subgraph中节点的总数
    graph_density = []  # 每个subgraph的图密度

    if args.dataset in ["PATTERN", "CLUSTER", "PascalVOC-SP", "COCO-SP"]:
        for iter, (g) in tqdm(enumerate(train_dataloader)):
            indices = torch.stack(g.edges())
            N = g.num_nodes()
            num_nodes.append(N)
            graph_density.append(g.num_edges() / (N * (N - 1)))
            A = dglsp.spmatrix(indices, shape=(N, N))
            if iter == 0:
                num_neigh = A.sum(1).int()
            else:
                num_neigh = torch.cat((num_neigh, A.sum(1).int()), 0)
    else:
        for iter, (g, _) in tqdm(enumerate(train_dataloader)):
            indices = torch.stack(g.edges())
            N = g.num_nodes()
            num_nodes.append(N)
            graph_density.append(g.num_edges() / (N * (N - 1)))
            A = dglsp.spmatrix(indices, shape=(N, N))
            if iter == 0:
                num_neigh = A.sum(1).int()
            else:
                num_neigh = torch.cat((num_neigh, A.sum(1).int()), 0)
        # print(num_neigh)
        # print(num_nodes)
    print("dataset ", args.dataset)
    print("# of subgraphs", len(num_nodes))

    num_nodes = torch.tensor(num_nodes)
    graph_density = torch.tensor(graph_density)

    print(
        f"# of nodes per subgraph mean {torch.mean(num_nodes.float())} std {torch.std(num_nodes.float())}"
    )
    print(
        f"# of neigh mean {torch.mean(num_neigh.float())} std {torch.std(num_neigh.float())}"
    )

    figure_boxplot(
        args.dataset,
        "num_nodes",
        num_nodes,
        torch.mean(num_nodes.float()),
        torch.std(num_nodes.float()),
    )
    figure_boxplot(
        args.dataset,
        "num_neigh",
        num_neigh,
        torch.mean(num_neigh.float()),
        torch.std(num_neigh.float()),
    )
    figure_boxplot(
        args.dataset,
        "graph_density",
        graph_density,
        torch.mean(graph_density.float()),
        torch.std(graph_density.float()),
    )


def full_graph_statistics(args):
    dataset = load_data_full_graph(args.dataset, "/workspace2/dataset")
    g = dataset[0]
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    graph_density = g.num_edges() / (N * (N - 1))
    A = dglsp.spmatrix(indices, shape=(N, N))
    num_neigh = A.sum(1).int()

    print("dataset ", args.dataset)
    print("# of nodes", N)
    print("graph density", graph_density)

    print(
        f"# of neigh mean {torch.mean(num_neigh.float())} std {torch.std(num_neigh.float())}"
    )
    figure_boxplot(
        args.dataset,
        "num_neigh",
        num_neigh,
        torch.mean(num_neigh.float()),
        torch.std(num_neigh.float()),
    )


def plot_dataset_perf(args):
    # formats = ["hyper", "indegree", "csr", "softmax"]
    # formats_label = [
    #     "All fuse(hyper)",
    #     "All fuse(indegree)",
    #     "All fuse(csr)",
    #     "Fuse softmax&SPMM",
    # ]
    # if args.conv == "dotgat":
    #     formats = ["hyper", "tile"]
    #     formats_label = ["All fuse(hyper)", "All fuse(tile`)"]
    if args.conv == "gt":
        formats = ["hyper", "csr", "softmax", "hybrid"]
        formats_label = [
            "All fuse(hyper)",
            "All fuse(csr)",
            "Fuse softmax&SPMM",
            "No fuse(hybrid format)",
        ]
    else:
        formats = ["hyper", "csr", "softmax"]
        formats_label = [
            "All fuse(hyper)",
            "All fuse(csr)",
            "Fuse softmax&SPMM",
        ]
    batch_sizes = [str(2**i) for i in range(4, 13)]

    time_no_fuse_all = []
    time_fuse_all = []

    result_dir = os.path.join(
        "/workspace2/fuse_attention", "dataset", args.dataset, args.conv
    )

    for i, format in enumerate(formats):
        a2 = []
        a3 = []
        for bs in batch_sizes:
            output = os.path.join(
                result_dir, f"{format}_dim{args.dim}_bs{bs}_result.pkl"
            )
            with open(output, "rb") as f:
                time_no_fuse, time_fuse = pickle.load(f)
            a2.append(mean(time_no_fuse))
            a3.append(mean(time_fuse))
        time_no_fuse_all.append(a2)
        time_fuse_all.append(a3)

    save_folder = f"/workspace2/fuse_attention/figure/dataset/{args.conv}"
    mkdir(save_folder)
    save_dir = os.path.join(save_folder, f"dim{args.dim}_{args.dataset}")
    title = f"""{args.dataset} dataset, dim={args.dim}"""

    # plot elapsed time of diff methods (log coordinate)
    fig = plt.figure(dpi=100, figsize=[12, 8])
    plt.ylabel("Elapsed time", fontsize=20)
    plt.xlabel("Batch Size", fontsize=20)
    for i, label in enumerate(formats_label):
        plt.plot(batch_sizes, time_fuse_all[i], "o-", label=label)
    plt.plot(batch_sizes, time_no_fuse_all[0], "o-", label="No fuse(single format)")
    plt.yscale("log")
    plt.title(title, fontsize=20)
    plt.legend()
    plt.savefig(save_dir + "_logtime.png")

    # plot elapsed time of diff methods (standard coordinate)
    fig = plt.figure(dpi=100, figsize=[12, 8])
    plt.ylabel("Elapsed time", fontsize=20)
    plt.xlabel("Batch Size", fontsize=20)
    for i, label in enumerate(formats_label):
        plt.plot(batch_sizes, time_fuse_all[i], "o-", label=label)
    plt.plot(batch_sizes, time_no_fuse_all[0], "o-", label="No fuse(single format)")
    plt.title(title, fontsize=20)
    plt.legend()
    plt.savefig(save_dir + "_time.png")

    # plot speedup of diff methods (standard coordinate)
    fig = plt.figure(dpi=100, figsize=[12, 8])
    plt.ylabel("Speedup", fontsize=20)
    plt.xlabel("Batch Size", fontsize=20)
    for i, label in enumerate(formats_label):
        speedup_mean = np.array(time_no_fuse_all[0]) / np.array(time_fuse_all[i])
        plt.plot(batch_sizes, speedup_mean, "o-", label=label)
    plt.xticks(batch_sizes)
    plt.title(title, fontsize=20)
    plt.ylim(bottom=1)
    plt.legend()
    plt.savefig(save_dir + "_speedup.png")


if __name__ == "__main__":
    # Load graph data
    parser = argparse.ArgumentParser(description="GT")
    args = parser_argument(parser)
    plot_dataset_perf(args)

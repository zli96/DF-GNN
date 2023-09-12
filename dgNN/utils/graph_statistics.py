import os

import pdb
import sys

import dgl.sparse as dglsp
import matplotlib.pyplot as plt
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))
print(os.path.join(current_dir, ".."))


import argparse

from tqdm import tqdm
from util import load_data_batch, load_data_full_graph


def figure_boxplot(dataset_name, fig_name, data, mean, std):
    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(data)
    plt.title(
        f"{dataset_name} {fig_name}, mean {mean:.2f}, std {std:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_{fig_name}.png")


def batch_graph_statistics(args):
    train_dataloader, _ = load_data_batch(args.dataset, 1, "/workspace2/dataset")

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


if __name__ == "__main__":
    # Load graph data
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    args = parser.parse_args()

    full_graphs = ["cora", "arxiv", "pubmed", "cite"]
    if args.dataset not in full_graphs:
        batch_graph_statistics(args)
    else:
        full_graph_statistics(args)

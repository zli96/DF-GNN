import matplotlib.pyplot as plt
import numpy as np
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from ogb.lsc import DglPCQM4Mv2Dataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset


def load_data_batch(dataset_name, batch_size):
    if dataset_name == "PCQM4Mv2-full":
        dataset = DglPCQM4Mv2Dataset(root="./data")
    else:
        dataset = DglGraphPropPredDataset(dataset_name, "./data/OGB")
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    train_dataloader = GraphDataLoader(
        dataset[train_idx], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl
    )
    return train_dataloader


def load_data_full_graph(dataset_name):
    if dataset_name == "cora":
        dataset = CoraGraphDataset()
    elif dataset_name == "arxiv":
        dataset = DglNodePropPredDataset("ogbn-arxiv")[0]
    elif dataset_name == "cite":
        dataset = CiteseerGraphDataset()
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset()
    else:
        raise ValueError(f"Unsupport dataset {dataset_name}")
    return dataset


def figure_num_std(
    dataset_name, batch_size, num, num_avg, num_edges, num_edges_avg, std, std_avg
):
    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(num)
    plt.title(
        f"{dataset_name} # of nodes per batch, mean {num_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_num_nodes_{batch_size}.png")
    fig.clear()

    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(num_edges)
    plt.title(
        f"{dataset_name} # of edges per batch, mean {num_edges_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_num_edges_{batch_size}.png")
    fig.clear()

    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(std)
    plt.title(
        f"{dataset_name} std of nodes per batch, mean {std_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_std_nodes_{batch_size}.png")


def figure_num_neigh_dist(dataset_name, num_neigh):
    max_neigh = 0
    for i in range(len(num_neigh) - 1, 0, -1):
        if num_neigh[i] != 0:
            max_neigh = i
            break
    num_neigh = num_neigh[:max_neigh]
    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.bar(range(len(num_neigh)), num_neigh)
    plt.title(
        f"{dataset_name} # of neighbors distribution",
        fontdict={"fontsize": 20},
    )
    for x1, y1 in enumerate(num_neigh):
        plt.text(x1, y1 + 10, y1.item(), ha="center", fontsize=16)
    plt.savefig(f"figure/{dataset_name}_num_neigh_dist.png")
    fig.clear()


def figure_nodes_neigh(dataset_name, num_neigh_per_node):
    fig = plt.figure()
    plt.plot(num_neigh_per_node, color="red")
    plt.title(
        f"{dataset_name} # of neighbors per nodes",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_nodes_neigh.png")
    fig.clear()

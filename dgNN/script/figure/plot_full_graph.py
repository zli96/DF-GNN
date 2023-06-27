import torch
from tqdm import tqdm
import argparse

from dgNN.utils import load_data_full_graph, figure_nodes_neigh

if __name__ == "__main__":
    # load dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    args = parser.parse_args()
    dataset = load_data_full_graph(args.dataset)
    g = dataset[0]
    num_neigh_g = g.adj().sum(dim=1).int()
    figure_nodes_neigh(args.dataset, num_neigh_g)
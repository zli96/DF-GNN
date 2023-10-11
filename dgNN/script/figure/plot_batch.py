import argparse

import torch

from dgl.dataloading import GraphDataLoader

from dgNN.utils import figure_num_neigh_dist, load_dataset_fn
from tqdm import tqdm


if __name__ == "__main__":

    # load dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    args = parser.parse_args()
    dataset, train_fn, collate_fn = load_dataset_fn(args.dataset, args.bs)
    train_dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    num_nodes_batch = []
    num_edges_batch = []
    std_nodes_batch = []
    num_neigh = torch.zeros((64)).int()
    for batch, (batched_g, _) in tqdm(enumerate(train_dataloader)):
        # num_nodes_batch.append(batched_g.num_nodes())
        # num_edges_batch.append(batched_g.num_edges())
        # std_nodes_batch.append(torch.std(batched_g.batch_num_nodes().float()))
        num_neigh_g = batched_g.adj().sum(dim=1).int()
        for i in num_neigh_g:
            num_neigh[i] += 1
    # num_nodes_avg = np.mean(num_nodes_batch)
    # num_edges_avg = np.mean(num_edges_batch)
    # std_nodes_avg = np.mean(std_nodes_batch)

    # figure_num_std(
    #     args.dataset,
    #     args.bs,
    #     num_nodes_batch,
    #     num_nodes_avg,
    #     num_edges_batch,
    #     num_edges_avg,
    #     std_nodes_batch,
    #     std_nodes_avg,
    # )
    figure_num_neigh_dist(args.dataset, num_neigh)

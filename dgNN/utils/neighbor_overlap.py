import argparse
import os
import pdb
import sys
from collections import Counter

import dgl.sparse as dglsp
import matplotlib.pyplot as plt
import torch
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from util import load_data_full_graph, load_dataset_fn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))
print(os.path.join(current_dir, ".."))


def counter(arr):
    return Counter(arr)


def figure_boxplot(args, fig_name, data, mean, std):
    dataset_name = args.dataset
    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(data)
    plt.title(
        f"{dataset_name} {fig_name}, mean {mean:.2f}, std {std:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(
        f"figure/neighbor_overlap/{fig_name}_{dataset_name}_blk{args.blocksize}.png"
    )


def check_smem(smem_limit):
    # if smem_limit > 12*1024:
    #     print("smem need more than 48KB")
    # else:
    #     return True
    if smem_limit > 16 * 1024:
        print("fail, smem need more than 64KB")
        return False
    return True


def cal_overlap_adj_nodes(A, num_nodes, blocksize, f):
    row_ptr, col_ind, val_idx = A.csr()
    overlap_coefficients = []  # optimized global memory access percent
    smem_demand = []  # smem needed for each thread block
    avg_neigh_times = []
    num_neighs = []
    num_unique_neighs = []
    reuse_time = []  # each neighbor reuse time
    for i in tqdm(range((num_nodes + blocksize - 1) // blocksize)):
        lb = i * blocksize
        hb = min(lb + blocksize, num_nodes)
        neighbors = col_ind[row_ptr[lb] : row_ptr[hb]]
        # print("num of neighbors", neighbors.numel())
        # print("num of unique neighbors", neighbors.unique().numel())
        # print("overlap coefficient", neighbors.unique().numel()/neighbors.numel())
        num_neigh = neighbors.numel()
        num_unique_neigh = neighbors.unique().numel()
        overlap_coefficients.append(neighbors.unique().numel() / neighbors.numel())
        num_neighs.append(num_neigh)
        num_unique_neighs.append(num_unique_neigh)

        neigh_time = list(counter(neighbors.tolist()).values())
        avg_neigh_time = sum(neigh_time) / len(neigh_time)
        avg_neigh_times.append(avg_neigh_time)
        reuse_time += neigh_time

        # pdb.set_trace()
        smem_limit = 2 * num_unique_neigh * f + num_neigh
        smem_demand.append(smem_limit)
        if not check_smem(smem_limit):
            print("num of neighbors", neighbors.numel())
            print("num of unique neighbors", neighbors.unique().numel())

    print(f"max smem consumption {max(smem_demand) * 4} bytes")
    print("neighbor reuse time: ", counter(reuse_time))
    print("all block pass")
    return overlap_coefficients, avg_neigh_times, num_neighs, num_unique_neighs


def neigh_overlap_adj_nodes(args):
    blocksize = args.blocksize
    dataset, train_fn, collate_fn = load_dataset_fn(
        args.dataset, args.bs, "/workspace2/dataset"
    )
    train_dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    overlap_coe = []
    if args.dataset == "PATTERN" or args.dataset == "CLUSTER":
        for iter, (g) in enumerate(train_dataloader):
            indices = torch.stack(g.edges())
            N = g.num_nodes()
            A = dglsp.spmatrix(indices, shape=(N, N))
            (
                overlap_coe,
                avg_neigh_times,
                num_neighs,
                num_unique_neighs,
            ) = cal_overlap_adj_nodes(A, N, blocksize, args.dim)
            break
    else:
        for iter, (g, _) in enumerate(train_dataloader):
            indices = torch.stack(g.edges())
            N = g.num_nodes()
            A = dglsp.spmatrix(indices, shape=(N, N))
            (
                overlap_coe,
                avg_neigh_times,
                num_neighs,
                num_unique_neighs,
            ) = cal_overlap_adj_nodes(A, N, blocksize, args.dim)
            break

    overlap_coe = torch.tensor(overlap_coe)
    avg_neigh_times = torch.tensor(avg_neigh_times)
    num_neighs = torch.tensor(num_neighs)
    num_unique_neighs = torch.tensor(num_unique_neighs)

    figure_boxplot(
        args,
        "memory_reduce",
        overlap_coe,
        torch.mean(overlap_coe.float()),
        torch.std(overlap_coe.float()),
    )
    figure_boxplot(
        args,
        "avg_neighbor_reuse",
        avg_neigh_times,
        torch.mean(avg_neigh_times.float()),
        torch.std(avg_neigh_times.float()),
    )

    figure_boxplot(
        args,
        "num_neighs",
        num_neighs,
        torch.mean(num_neighs.float()),
        torch.std(num_neighs.float()),
    )
    figure_boxplot(
        args,
        "num_unique_neighs",
        num_unique_neighs,
        torch.mean(num_unique_neighs.float()),
        torch.std(num_unique_neighs.float()),
    )


if __name__ == "__main__":
    # Load graph data
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--bs", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--blocksize", type=int, default=8)

    args = parser.parse_args()

    neigh_overlap_adj_nodes(args)

    # full_graphs = ["cora", "arxiv", "pubmed", "cite"]
    # if args.dataset not in full_graphs:
    #     batch_graph_statistics(args)
    # else:
    #     full_graph_statistics(args)

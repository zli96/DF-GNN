import argparse

import dgl.sparse as dglsp

import torch
from dgl.dataloading import GraphDataLoader

from dgNN.layers import choose_GTlayer, GATConv_dgNN, GATConv_hyper

from dgNN.layers.util import preprocess_Hyper
from dgNN.utils import load_dataset_fn, parser_argument, train_profile


def preprocess_GAT(g, **args):
    row_ptr, col_ind, _ = g.adj_tensors("csr")
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    M = g.num_edges()
    val = torch.ones(M) * 1.5
    A = dglsp.spmatrix(indices, val=val, shape=(N, N))
    return A, row_ptr.int(), col_ind.int()


def main(args):

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset, train_fn, collate_fn = load_dataset_fn(args.dataset, args.data_dir)
    if args.format == "csr":
        layer = GATConv_dgNN
        preprocess_func = preprocess_GAT
    elif args.format == "hyper":
        layer = GATConv_hyper
        preprocess_func = preprocess_Hyper
    else:
        raise ValueError(f"Unsupported format {args.format}")
    GTlayer = choose_GTlayer(
        args.dataset, MHAlayer=layer, hidden_size=args.dim, num_heads=args.heads
    )
    GTlayer = GTlayer.to(dev)
    train_dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    if args.profile:
        train_profile(
            preprocess_func, GTlayer, train_dataloader, dev, args.dataset, dim=args.dim
        )
    else:
        time_no_fuse, time_fuse = train_fn(
            preprocess_func, GTlayer, train_dataloader, dev, dim=args.dim
        )
        print("----------------------Result------------------------")
        print(
            "no-fuse average time {:.4f} ms".format(
                sum(time_no_fuse[:-1]) / (len(time_no_fuse) - 1)
            )
        )
        print(
            "fuse average time {:.4f} ms".format(
                sum(time_fuse[:-1]) / (len(time_fuse) - 1)
            )
        )
        print(sum(time_no_fuse[:-1]) / (len(time_no_fuse) - 1))
        print(sum(time_fuse[:-1]) / (len(time_fuse) - 1))


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="GAT")
    args = parser_argument(parser)

    main(args)

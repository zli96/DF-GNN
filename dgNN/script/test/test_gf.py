import argparse

import torch

from dgl.dataloading import GraphDataLoader

from dgNN.layers import (
    choose_GTlayer,
    SparseMHA,
    SparseMHA_hyper,
    SparseMHA_hyper_nofuse,
    SparseMHA_outdegree,
    SparseMHA_subgraph,
)
from dgNN.utils import (
    load_dataset_fn,
    parser_argument,
    preprocess_CSR,
    preprocess_Hyper,
    preprocess_Outdegree,
    preprocess_SubGraph,
    subgraph_filter,
)

if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="GF")
    args = parser_argument(parser)

    if args.format == "csr":
        layer = SparseMHA
        preprocess_func = preprocess_CSR
    elif args.format == "hyper":
        layer = SparseMHA_hyper
        preprocess_func = preprocess_Hyper
    elif args.format == "hyper_nofuse":
        layer = SparseMHA_hyper_nofuse
        preprocess_func = preprocess_Hyper
    elif args.format == "outdegree":
        layer = SparseMHA_outdegree
        preprocess_func = preprocess_Outdegree
    elif args.format == "subgraph":
        layer = SparseMHA_subgraph
        preprocess_func = preprocess_SubGraph
    else:
        raise ValueError(f"Unsupported format {args.format}")

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    GTlayer = choose_GTlayer(
        args.dataset, MHAlayer=layer, hidden_size=args.dim, num_heads=args.heads
    )
    GTlayer = GTlayer.to(dev)
    print("GTlayer", GTlayer)
    # load dataset
    dataset, train_fn, collate_fn = load_dataset_fn(args.dataset, args.data_dir)
    if args.subgraph_filter:
        dataset = subgraph_filter(dataset, args.dataset, args.dim, args.heads)
        train_dataloader = GraphDataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        if args.format == "subgraph":
            raise ValueError(
                "subgraph method only supporded when args.subgraph_filter is True"
            )
        train_dataloader = GraphDataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

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
        "fuse average time {:.4f} ms".format(sum(time_fuse[:-1]) / (len(time_fuse) - 1))
    )
    print(sum(time_no_fuse[:-1]) / (len(time_no_fuse) - 1))
    print(sum(time_fuse[:-1]) / (len(time_fuse) - 1))

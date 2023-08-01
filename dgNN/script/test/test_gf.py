import argparse

import torch

from dgNN.layers import (
    choose_GTlayer,
    SparseMHA,
    SparseMHA_ELL,
    SparseMHA_hyper,
    SparseMHA_subgraph,
)
from dgNN.utils import (
    load_data_batch,
    parser_argument,
    preprocess_CSR,
    preprocess_ELL,
    preprocess_Hyper,
    preprocess_SubGraph,
    train,
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
    elif args.format == "subgraph":
        layer = SparseMHA_subgraph
        preprocess_func = preprocess_SubGraph
    else:
        raise ValueError(f"Unsupported format {args.format}")

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_dataloader, train_fn = load_data_batch(
        args.dataset, args.batch_size, args.data_dir
    )
    GTlayer = choose_GTlayer(
        args.dataset, MHAlayer=layer, hidden_size=args.dim, num_heads=args.heads
    )
    GTlayer = GTlayer.to(dev)
    print("GTlayer", GTlayer)
    time_no_fuse, time_fuse = train_fn(preprocess_func, GTlayer, train_dataloader, dev)

    print("----------------------Result------------------------")
    print(
        "no-fuse average time {:.4f} ms".format(
            sum(time_no_fuse[:-1]) / (len(time_no_fuse) - 1)
        )
    )
    print(
        "fuse average time {:.4f} ms".format(sum(time_fuse[:-1]) / (len(time_fuse) - 1))
    )

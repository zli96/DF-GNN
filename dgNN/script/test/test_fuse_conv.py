import argparse

import dgl.sparse as dglsp

import torch
from dgl.dataloading import GraphDataLoader

from dgNN.layers import (
    choose_Model,
    load_layer_DOTGAT,
    load_layer_GAT,
    load_layer_GT,
    load_prepfunc,
)
from dgNN.utils import load_dataset_fn, parser_argument, train_profile


def main(args):

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init model
    dataset, train_fn, collate_fn = load_dataset_fn(args.dataset, args.data_dir)
    if args.conv == "dotgat":
        layer = load_layer_DOTGAT(args)
    elif args.conv == "gat":
        layer = load_layer_GAT(args)
    elif args.conv == "gt":
        layer = load_layer_GT(args)
    else:
        raise ValueError(f"unknown graph conv {args.conv}")

    preprocess_func = load_prepfunc(args)
    model = choose_Model(
        args.dataset, MHAlayer=layer, hidden_size=args.dim, num_heads=args.heads
    )
    model = model.to(dev)
    print("model", model)

    train_dataloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # profile mode
    if args.profile:
        fuse_flag = args.format != "nofuse"
        train_profile(
            preprocess_func,
            model,
            train_dataloader,
            dev,
            args.dataset,
            fuse_flag,
            dim=args.dim,
        )
    # normal run
    else:
        time_no_fuse, time_fuse = train_fn(
            preprocess_func, model, train_dataloader, dev, dim=args.dim
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
    parser = argparse.ArgumentParser(description="DOTGAT")
    args = parser_argument(parser)
    main(args)

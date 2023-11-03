import argparse

import torch
from dgl.dataloading import GraphDataLoader

from dgNN.layers import choose_GTlayer, load_layer_GAT, load_prepfunc
from dgNN.utils import load_dataset_fn, parser_argument, train_profile


def main(args):

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init model
    dataset, train_fn, collate_fn = load_dataset_fn(args.dataset, args.data_dir)
    layer = load_layer_GAT(args)
    preprocess_func = load_prepfunc(args)
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

    # profile mode
    if args.profile:
        train_profile(
            preprocess_func, GTlayer, train_dataloader, dev, args.dataset, dim=args.dim
        )
    # normal run
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

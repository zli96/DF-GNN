import argparse
import pdb

import torch

from dgl.dataloading import GraphDataLoader

from dgNN.layers import choose_GTlayer, SparseMHA_subgraph
from dgNN.utils import load_data_batch, preprocess_Outdegree


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="/workspace2/dataset")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    if args.profile:
        print("--------------profile mode------------------------")
    print("Dataset", args.dataset)
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_dataloader, train_fn = load_data_batch(
        args.dataset, args.batch_size, args.data_dir
    )

    GTlayer = choose_GTlayer(
        args.dataset,
        MHAlayer=SparseMHA_subgraph,
        hidden_size=args.dim,
        num_heads=args.heads,
    )
    GTlayer = GTlayer.to(dev)
    dataset_name = args.dataset

    print("GTlayer", GTlayer)

    if args.profile:
        time_no_fuse, time_fuse = train_fn(
            preprocess_Outdegree,
            GTlayer,
            train_dataloader,
            dev,
            fuse=True,
            dim=args.dim,
        )
    else:
        time_no_fuse, time_fuse = train_fn(
            preprocess_Outdegree, GTlayer, train_dataloader, dev, dim=args.dim
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

import argparse
import os, pickle

import torch

from dgl.dataloading import GraphDataLoader

from dgNN.layers import choose_Model, load_layer_GT, load_prepfunc, subgraph_filter
from dgNN.utils import load_dataset_fn, mkdir, parser_argument, train_profile


def main(args):
    layer = load_layer_GT(args)
    preprocess_func = load_prepfunc(args)

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = choose_Model(
        args.dataset, MHAlayer=layer, hidden_size=args.dim, num_heads=args.heads
    )
    model = model.to(dev)
    print("model", model)
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
    if args.profile:
        train_profile(
            preprocess_func, model, train_dataloader, dev, args.dataset, dim=args.dim
        )
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

        if args.store_result:
            result_dir = os.path.join(
                "/workspace2/fuse_attention", "dataset", args.dataset
            )
            mkdir(result_dir)
            with open(
                os.path.join(
                    result_dir,
                    f"{args.format}_dim{args.dim}_bs{args.batch_size}_result.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump([time_no_fuse[:-1], time_fuse[:-1]], f)
                print("-----------dump run result--------------------")


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="GT")
    args = parser_argument(parser)
    main(args)

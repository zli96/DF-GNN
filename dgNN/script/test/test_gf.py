import argparse

import torch

from dgNN.layers import GTlayer_mol, SparseMHA, SparseMHA_ELL, SparseMHA_hyper
from dgNN.utils import (
    load_data_batch,
    parser_argument,
    preprocess_CSR,
    preprocess_ELL,
    preprocess_Hyper,
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
    else:
        raise ValueError(f"Unsupported format {args.format}")

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    train_dataloader = load_data_batch(args.dataset, args.batch_size, args.data_dir)
    GTlayer = GTlayer_mol(layer=layer, hidden_size=args.dim, num_heads=args.heads).to(
        dev
    )
    time_no_fuse, time_fuse = train(preprocess_func, GTlayer, train_dataloader, dev)

    print("----------------------Result------------------------")
    print(
        "no-fuse average time {:.4f} ms".format(
            sum(time_no_fuse[:-1]) / (len(time_no_fuse) - 1)
        )
    )
    print(
        "fuse average time {:.4f} ms".format(sum(time_fuse[:-1]) / (len(time_fuse) - 1))
    )

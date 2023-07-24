import argparse
import pdb

import torch
import torch.nn.functional as F
from dgNN.layers import GTlayer_mol, SparseMHA, SparseMHA_ELL, SparseMHA_hyper
from dgNN.utils import (
    load_data_batch,
    parser_argument,
    preprocess_CSR,
    preprocess_ELL,
    preprocess_Hyper,
    train_profile,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GF")
    args = parser_argument(parser)

    if args.format == "csr":
        layer = SparseMHA
        preprocess_func = preprocess_CSR
        fuse = True
    elif args.format == "hyper":
        layer = SparseMHA_hyper
        preprocess_func = preprocess_Hyper
        fuse = True
    elif args.format == "nofuse":
        layer = SparseMHA
        preprocess_func = preprocess_CSR
        fuse = False
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
    train_profile(preprocess_func, GTlayer, train_dataloader, dev, fuse=fuse)

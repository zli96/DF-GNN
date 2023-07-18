import argparse

import torch
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader

from dgNN.layers import GTlayer_mol, SparseMHA, SparseMHA_ELL, SparseMHA_hyper
from dgNN.utils import preprocess_CSR, preprocess_ELL, preprocess_Hyper, train
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--format", type=str, default="csr")

    args = parser.parse_args()
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    print("format: ", args.format)

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
    dataset = AsGraphPredDataset(
        DglGraphPropPredDataset(f"{args.dataset}", f"{args.data_dir}")
    )
    evaluator = Evaluator(f"{args.dataset}")
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=args.batch_size,
        collate_fn=collate_dgl,
        shuffle=False,
    )

    out_size = dataset.num_tasks
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

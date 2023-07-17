import argparse
import pdb

import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from dgNN.layers import SparseMHA

from dgNN.utils import preprocess_CSR, train_profile
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=2, num_heads=1):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.atom_encoder = AtomEncoder(hidden_size)

    def forward(self, params, X, fuse=True):
        h = self.atom_encoder(X)
        h = self.MHA(params, h, fuse)
        return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    args = parser.parse_args()
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
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
    layer = GTLayer(hidden_size=args.dim, num_heads=args.heads).to(dev)
    train_profile(preprocess_CSR, layer, train_dataloader, dev)

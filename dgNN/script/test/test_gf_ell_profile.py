import argparse
import pdb

import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader

from dgNN.layers import SparseMHA_ELL
from dgNN.utils import load_data_full_graph, preprocess_ELL, train_profile
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=2, num_heads=1):
        super().__init__()
        self.MHA = SparseMHA_ELL(hidden_size=hidden_size, num_heads=num_heads)
        self.atom_encoder = AtomEncoder(hidden_size)

    def forward(self, params, X, ell=False):
        h = self.atom_encoder(X)
        h = self.MHA(params, h, ell)
        return h


col_part_config = {
    "arxiv": 1,
    "proteins": 8,
    "pubmed": 1,
    "citeseer": 1,
    "cora": 1,
    "ppi": 16,
    "reddit": 8,
    "products": 16,
}

bucketing_config = {
    "arxiv": [1, 2, 4, 8, 16, 32],
    "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "pubmed": [1, 2, 4, 8, 16, 32],
    "citeseer": [1, 2, 4],
    "cora": [1, 2, 4],
    "ppi": [1, 2, 4, 8, 16, 32],
    "products": [1, 2, 4, 8, 16, 32],
    "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument(
        "--dataset", "-d", type=str, default="cora", help="dataset name"
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")
    args = parser.parse_args()
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    name = args.dataset
    layer = GTLayer(hidden_size=args.dim, num_heads=args.heads).to(dev)
    dataset = AsGraphPredDataset(
        DglGraphPropPredDataset("ogbg-molhiv", f"{args.data_dir}")
    )
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=args.batch_size,
        collate_fn=collate_dgl,
        shuffle=False,
    )

    time_no_fuse, time_fuse = train_profile(
        preprocess_ELL,
        layer,
        train_dataloader,
        dev,
        bucket_sizes=bucketing_config[name],
        num_col_parts=col_part_config[name],
    )

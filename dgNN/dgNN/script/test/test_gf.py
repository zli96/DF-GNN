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
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=2, num_heads=1):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.atom_encoder = AtomEncoder(hidden_size)

    def forward(self, g, X, fuse=False):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        h = self.atom_encoder(X)
        h = self.MHA(A, h, fuse)

        return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = AsGraphPredDataset(DglGraphPropPredDataset("ogbg-molhiv", "./data/OGB"))
    evaluator = Evaluator("ogbg-molhiv")
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=args.batch_size,
        collate_fn=collate_dgl,
        shuffle=False,
    )

    out_size = dataset.num_tasks
    layer = GTLayer(hidden_size=args.dim, num_heads=args.heads).to(dev)
    time_no_fuse = []
    time_fuse = []

    for i, (batched_g, labels) in enumerate(train_dataloader):
        batched_g, labels = batched_g.to(dev), labels.to(dev)
        # print("----------------------without fuse--------------------------")
        logits, elapsed_time = layer(batched_g, batched_g.ndata["feat"])
        if i > 5:
            time_no_fuse.append(elapsed_time)
            print(f"epoch {i} non-fused time ", elapsed_time)
            # print("----------------------with fuse--------------------------")
            logits_fuse, elapsed_time = layer(batched_g, batched_g.ndata["feat"], fuse=True)
            time_fuse.append(elapsed_time)
            # pdb.set_trace()
            print(f"epoch {i} fused time ", elapsed_time)
            if all(torch.isclose(logits, logits_fuse, atol=0.001).flatten()):
                print("the results are the same, success!!!!!!!!!!")
            else:
                for i in range(logits.shape[0]):
                    if not all(torch.isclose(logits[i], logits_fuse[i], atol=0.001).flatten()):
                        print(f"error node {i} mismatch")
                        # print("neighbor nodes", col_ind[row_ptr[i]:row_ptr[i+1]])
                        print(logits[i])
                        print(logits_fuse[i])
                        pdb.set_trace()

            if i == 10:
                print("----------------------Result------------------------")
                print("no-fuse average time ", sum(time_no_fuse) / len(time_no_fuse))
                print("fuse average time ", sum(time_fuse) / len(time_fuse))
                exit()

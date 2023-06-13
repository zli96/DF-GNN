import argparse

import pdb

import dgl
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F


from dgNN.layers import SparseMHA
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset

class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, in_size, hidden_size, num_heads=1):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.linear = nn.Linear(in_size, hidden_size)

    def forward(self, A, X, fuse=False):
        h = self.linear(X)
        h = self.MHA(A, h, fuse)
        return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dataset", type=str, default="cora")
    args = parser.parse_args()
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    # Load graph from the existing dataset.
    if args.dataset == "cora":
        dataset = CoraGraphDataset()
    elif args.dataset == "arxiv":
        dataset = DglNodePropPredDataset("ogbn-arxiv")[0]
    elif args.dataset == "cite":
        dataset = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        dataset = PubmedGraphDataset()
    else:
        raise ValueError(f"Unsupport dataset {args.dataset}")
    g = dataset[0].to(dev)
    # Create the sparse adjacency matrix A.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Add self-loops.
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I

    X = g.ndata["feat"]
    in_size = X.shape[1]
    layer = GTLayer(in_size=in_size, hidden_size=args.dim, num_heads=args.heads).to(dev)

    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 1
    # iter = 10
    for epoch in range(50):
        # print("----------------------without fuse--------------------------")
        print(epoch)
        logits, elapsed_time = layer(A_hat, X)
        if epoch >= warmup:
            time_no_fuse.append(elapsed_time)
            print(f"epoch {epoch} non-fused time %.4f" % elapsed_time)
            # print("----------------------with fuse--------------------------")
            logits_fuse, elapsed_time = layer(
                A_hat, X, fuse=True
            )
            time_fuse.append(elapsed_time)
            # pdb.set_trace()
            print(f"epoch {epoch} fused time %.4f" % elapsed_time)
            if all(torch.isclose(logits, logits_fuse, atol=0.001).flatten()):
                print("the results are the same, success!!!!!!!!!!")
            else:
                for epoch in range(logits.shape[0]):
                    if not all(
                        torch.isclose(
                            logits[epoch], logits_fuse[epoch], atol=0.001
                        ).flatten()
                    ):
                        print(f"error node {epoch} mismatch")
                        # print("neighbor nodes", col_ind[row_ptr[epoch]:row_ptr[epoch+1]])
                        print(logits[epoch])
                        print(logits_fuse[epoch])
                        pdb.set_trace()

    print("----------------------Result------------------------")
    print(
        "no-fuse average time {:.4f} ms".format(sum(time_no_fuse) / len(time_no_fuse))
    )
    print("fuse average time {:.4f} ms".format(sum(time_fuse) / len(time_fuse)))

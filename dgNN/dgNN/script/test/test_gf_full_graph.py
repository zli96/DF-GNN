import argparse
import pdb

import dgl
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgNN.layers import SparseMHA
from dgNN.utils import load_data_full_graph


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
    parser.add_argument("--dataset", type=str, default="cora")
    args = parser.parse_args()
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = load_data_full_graph(args.dataset)
    g = dataset[0].to(dev)
    # Create the sparse adjacency matrix A.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Add self-loops.
    I = dglsp.identity(A.shape, device=dev)
    A_hat = A + I
    print("max neighbor is ", max(A_hat.sum(dim=1)))
    X = g.ndata["feat"]
    in_size = X.shape[1]
    layer = GTLayer(in_size=in_size, hidden_size=args.dim, num_heads=args.heads).to(dev)

    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 1
    # iter = 10
    for epoch in range(30):
        print(epoch)
        logits, elapsed_time = layer(A_hat, X)
        if epoch >= warmup:
            time_no_fuse.append(elapsed_time)
            print(f"epoch {epoch} non-fused time %.4f" % elapsed_time)
            logits_fuse, elapsed_time = layer(A_hat, X, fuse=True)
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

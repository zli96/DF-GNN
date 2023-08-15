import os

import pdb
import sys

import dgl
import dgl.sparse as dglsp
import torch

from dgNN.layers import GTlayer, SparseMHA, SparseMHA_hyper
from dgNN.utils import preprocess_CSR, preprocess_Hyper

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))
# print(os.path.join(current_dir, ".."))


import argparse


# generate graph with constant degree
def generate_graph_constant_degree(num_nodes, num_neigh):
    src = torch.arange(0, num_nodes).long()
    src = src.repeat(num_neigh, 1).T.flatten()
    dst = torch.randint_like(src, 0, num_nodes - 1)
    g = dgl.graph((src, dst), num_nodes=num_nodes)
    g.ndata["feat"] = torch.randn(num_nodes, 32)
    return g


if __name__ == "__main__":
    # Load graph data
    parser = argparse.ArgumentParser(description="Graph generator")
    parser.add_argument("--format", type=str, default="csr")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--avg-nodes", type=int, default=70)
    parser.add_argument("--num-neigh", type=int, default=32)
    args = parser.parse_args()

    num_nodes = args.batch_size * args.avg_nodes
    g = generate_graph_constant_degree(num_nodes, args.num_neigh)
    print(g)
    print("format", args.format)
    print("num_neigh", args.num_neigh)
    print("avg-nodes", args.avg_nodes)
    print("batch-size", args.batch_size)

    if args.format == "csr":
        layer = SparseMHA
        preprocess_func = preprocess_CSR
    elif args.format == "hyper":
        layer = SparseMHA_hyper
        preprocess_func = preprocess_Hyper
    else:
        raise ValueError(f"Unsupported format {args.format}")

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    g = g.to(dev)

    # Create the sparse adjacency matrix A.
    params = preprocess_func(g)
    params = [param.to(dev) for param in params]
    X = g.ndata["feat"]
    in_size = X.shape[1]
    GTlayer = GTlayer(layer, in_size=in_size, hidden_size=32, num_heads=1).to(dev)

    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 2
    for epoch in range(30):
        logits, elapsed_time = GTlayer(params, X)
        if epoch >= warmup:
            time_no_fuse.append(elapsed_time)
            print(f"epoch {epoch} non-fused time %.4f" % elapsed_time)
            logits_fuse, elapsed_time = GTlayer(params, X, fuse=True)
            time_fuse.append(elapsed_time)
            print(f"epoch {epoch} fused time %.4f" % elapsed_time)
            if epoch < 3:
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
                            print(logits[epoch])
                            print(logits_fuse[epoch])
                            pdb.set_trace()

    print("----------------------Result------------------------")
    print(
        "no-fuse average time {:.4f} ms".format(sum(time_no_fuse) / len(time_no_fuse))
    )
    print("fuse average time {:.4f} ms".format(sum(time_fuse) / len(time_fuse)))

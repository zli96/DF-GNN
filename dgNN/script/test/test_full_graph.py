import argparse
import os, pickle

import torch
import torch.nn as nn

from dgNN.layers import load_layer, load_prepfunc
from dgNN.utils import check_correct, load_data_full_graph, mkdir


class Model(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, MHAlayer, in_size, hidden_size):
        super().__init__()
        self.inproj = nn.Linear(in_size, hidden_size)
        self.MHA = MHAlayer

    def forward(self, params, X, fuse=False):
        h = self.inproj(X)
        h = self.MHA(params, h, fuse)
        return h


def PrintGraphStruct(g):
    print("----------graph statics -----------")
    print(f"# of nodes {g.num_nodes()}")
    print(f"# of edges {g.num_edges()}")
    print(f"avg. degree {torch.mean(g.out_degrees().float()):.2f}")
    print(f"max. degree {max(g.out_degrees())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="full_graph")
    parser.add_argument("--conv", type=str, default="gt")
    parser.add_argument("--format", type=str, default="csr")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--store-result", action="store_true")
    args = parser.parse_args()

    layer = load_layer(args)
    preprocess_func = load_prepfunc(args)

    print("GraphConv", args.conv)
    print("format:", args.format)
    print("dataset:", args.dataset)
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    if args.store_result:
        print("will store the pref result")

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = load_data_full_graph(args.dataset, args.data_dir)
    g = dataset[0].to(dev)
    PrintGraphStruct(g)

    # Create the sparse adjacency matrix A.
    params = preprocess_func(g)
    X = g.ndata["feat"]
    in_size = X.shape[1]
    model = Model(MHAlayer=layer, in_size=in_size, hidden_size=args.dim)
    model = model.to(dev)
    print("model", model)
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []

    print("------warmup------")
    for epoch in range(1):
        logits, elapsed_time = model(params, X)

    print("------inference------")
    for epoch in range(10):
        if epoch < 2:
            logits, elapsed_time_nofuse = model(params, X)
            time_no_fuse.append(elapsed_time_nofuse)
            logits_fuse, elapsed_time = model(params, X, fuse=True)
            time_fuse.append(elapsed_time)
            check_correct(logits[:1000], logits_fuse[:1000], params)
            check_correct(logits[-1000:], logits_fuse[-1000:], params)
            print(f"epoch {epoch} non-fused time %.4f" % elapsed_time_nofuse)
            print(f"epoch {epoch} fused time %.4f" % elapsed_time)
        else:
            logits_fuse, elapsed_time = model(params, X, fuse=True)
            time_fuse.append(elapsed_time)
            print(f"epoch {epoch} fused time %.4f" % elapsed_time)

    print("----------------------Result------------------------")
    print(
        "no-fuse average time {:.4f} ms".format(sum(time_no_fuse) / len(time_no_fuse))
    )
    print("fuse average time {:.4f} ms".format(sum(time_fuse) / len(time_fuse)))

    if args.store_result:
        result_dir = os.path.join(
            "/workspace2/fuse_attention", "dataset", args.dataset, args.conv
        )
        mkdir(result_dir)
        with open(
            os.path.join(
                result_dir,
                f"{args.format}_dim{args.dim}_result.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump([time_no_fuse[:-1], time_fuse[:-1]], f)
            print("-----------dump run result--------------------")

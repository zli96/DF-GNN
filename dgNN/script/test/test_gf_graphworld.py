import argparse

import os, pdb, pickle, torch

import dgl

from dgNN.layers import GTlayer, SparseMHA, SparseMHA_hyper, SparseMHA_outdegree
from dgNN.utils import (
    check_correct,
    Move2Device,
    preprocess_CSR,
    preprocess_Hyper,
    preprocess_Outdegree,
)


def train(process_func, layer, dev, args, **kwargs):
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    avg_degrees = []
    warmup = 1
    for g_id in range(warmup):
        try:
            with open(os.path.join(args.output, f"{g_id}.pkl"), "rb") as f:
                g = pickle.load(f)
            params = process_func(g, **kwargs)
            g, params = Move2Device([g, params], dev)
            logits, elapsed_time = layer(params, g.ndata["feat"])
        except IOError:
            print("---------read INPUT file fail!!!--------------")
            exit()

    for g_id in range(args.graph_range):
        try:
            with open(os.path.join(args.output, f"{g_id}_config.pkl"), "rb") as f:
                config = pickle.load(f)
                print(config)
            with open(os.path.join(args.output, f"{g_id}.pkl"), "rb") as f:
                g = pickle.load(f)
            print("graph id", g_id)
            avg_degree = 2 * g.num_edges() / g.num_nodes()
            print("avg degree", avg_degree)
            print("in degree", torch.mean(g.in_degrees().float()).item())
            avg_degrees.append(avg_degree)
            params = process_func(g, **kwargs)
            g, params = Move2Device([g, params], dev)
            logits, elapsed_time = layer(params, g.ndata["feat"])
            print(f"epoch {g_id} non-fused time %.4f" % elapsed_time)
            time_no_fuse.append(elapsed_time)
            # print("----------------------with fuse--------------------------")
            logits_fuse, elapsed_time = layer(params, g.ndata["feat"], fuse=True)
            time_fuse.append(elapsed_time)
            # pdb.set_trace()
            print(f"epoch {g_id} fused time %.4f" % elapsed_time)
            if g_id == 0:
                check_correct(logits[:10000], logits_fuse[:10000], params)
            print("--------------------------------------")
        except IOError:
            break

    with open(
        os.path.join(args.output, f"{args.format}_{args.dim}_result.pkl"), "wb"
    ) as f:
        pickle.dump([avg_degrees, time_no_fuse, time_fuse], f)

    return time_no_fuse, time_fuse


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--output", type=str)
    parser.add_argument("--graph-range", type=int, default=100)
    parser.add_argument("--format", type=str, default="csr")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")
    parser.add_argument("--rerun", action="store_true")

    args = parser.parse_args()

    print("format", args.format)
    print("hidden dim", args.dim)
    print("output", args.output)
    print("graph-range", args.graph_range)

    if args.format == "csr":
        layer = SparseMHA
        preprocess_func = preprocess_CSR
    elif args.format == "hyper":
        layer = SparseMHA_hyper
        preprocess_func = preprocess_Hyper
    elif args.format == "outdegree":
        layer = SparseMHA_outdegree
        preprocess_func = preprocess_Outdegree
    else:
        raise ValueError(f"Unsupported format {args.format}")

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    layer = GTlayer(layer, 16, args.dim, args.heads)
    layer = layer.to(dev)
    print("GTlayer", layer)

    if args.rerun or not os.path.exists(
        os.path.join(args.output, f"{args.format}_result.pkl")
    ):
        time_no_fuse, time_fuse = train(preprocess_func, layer, dev, args, dim=args.dim)
    else:
        with open(os.path.join(args.output, f"{args.format}_result.pkl"), "rb") as f:
            _, time_no_fuse, time_fuse = pickle.load(f)

    print("----------------------Result------------------------")
    print(
        "no-fuse average time {:.4f} ms".format(sum(time_no_fuse) / (len(time_no_fuse)))
    )
    print("fuse average time {:.4f} ms".format(sum(time_fuse) / (len(time_fuse))))
    print(sum(time_no_fuse) / (len(time_no_fuse)))
    print(sum(time_fuse) / (len(time_fuse)))

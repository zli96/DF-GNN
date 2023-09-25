import argparse

import os, pdb, pickle, torch

import dgl

from dgNN.layers import GTlayer, SparseMHA, SparseMHA_hyper, SparseMHA_subgraph
from dgNN.utils import (
    check_correct,
    preprocess_CSR,
    preprocess_Hyper,
    preprocess_SubGraph,
)


def g_pkl2DGL(graph):
    num_vertex, edge_index, node_feature = graph
    # print("num_vertex", num_vertex)
    # print(edge_index.shape)
    # print(node_feature.shape)
    # print(node_feature.dtype)
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_vertex)
    g.ndata["feat"] = node_feature.to(torch.float)
    return g


def train(process_func, layer, dev):
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 1
    for g_id in range(warmup):
        try:
            with open(os.path.join(args.output, f"{g_id}.pkl"), "rb") as f:
                graph_pkl = pickle.load(f)
                g = g_pkl2DGL(graph_pkl)
            params = process_func(g)
            if params == None:
                continue
            params = [param.to(dev) for param in params]
            g = g.to(dev)
            logits, elapsed_time = layer(params, g.ndata["feat"])
        except IOError:
            break
    for g_id in range(args.graph_range):
        try:
            with open(os.path.join(args.output, f"{g_id}_config.pkl"), "rb") as f:
                config = pickle.load(f)
                print(config)
            with open(os.path.join(args.output, f"{g_id}.pkl"), "rb") as f:
                graph_pkl = pickle.load(f)
                g = g_pkl2DGL(graph_pkl)
            print("graph id", g_id)
            params = process_func(g)
            if params == None:
                continue
            params = [param.to(dev) for param in params]
            g = g.to(dev)
            logits, elapsed_time = layer(params, g.ndata["feat"])
            print(f"epoch {g_id} non-fused time %.4f" % elapsed_time)
            time_no_fuse.append(elapsed_time)
            # print("----------------------with fuse--------------------------")
            logits_fuse, elapsed_time = layer(params, g.ndata["feat"], fuse=True)
            time_fuse.append(elapsed_time)
            # pdb.set_trace()
            print(f"epoch {g_id} fused time %.4f" % elapsed_time)
            if g_id < 3:
                check_correct(logits, logits_fuse, params)
            if g_id == 20:
                break
            print("--------------------------------------")
        except IOError:
            break

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
    elif args.format == "subgraph":
        layer = SparseMHA_subgraph
        preprocess_func = preprocess_SubGraph
    else:
        raise ValueError(f"Unsupported format {args.format}")

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    layer = GTlayer(layer, 16, args.dim, args.heads)
    layer = layer.to(dev)
    print("GTlayer", layer)
    time_no_fuse, time_fuse = train(preprocess_func, layer, dev)

    print("----------------------Result------------------------")
    print(
        "no-fuse average time {:.4f} ms".format(sum(time_no_fuse) / (len(time_no_fuse)))
    )
    print("fuse average time {:.4f} ms".format(sum(time_fuse) / (len(time_fuse))))
    print(sum(time_no_fuse) / (len(time_no_fuse)))
    print(sum(time_fuse) / (len(time_fuse)))

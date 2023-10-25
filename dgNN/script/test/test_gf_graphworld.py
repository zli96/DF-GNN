import argparse

import os, pdb, pickle, torch, warnings

import dgl

from dgNN.layers import GTlayer, load_layer_prepfunc
from dgNN.utils import check_correct, Move2Device


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
            num_neigh = g.out_degrees().float()
            print("num neigh mean", torch.mean(num_neigh).item())
            print("num neigh std", torch.std(num_neigh).item())
            if any(num_neigh == 0):
                warnings.warn("exit zero-neighbor node")
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
    if args.store_result:
        with open(
            os.path.join(args.output, f"{args.format}_{args.dim}_result.pkl"), "wb"
        ) as f:
            pickle.dump([avg_degrees, time_no_fuse, time_fuse], f)
        print("-----------dump run result--------------------")

    return time_no_fuse, time_fuse


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--output", type=str)
    parser.add_argument("--graph-range", type=int, default=20)
    parser.add_argument("--format", type=str, default="csr")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--store-result", action="store_true")

    args = parser.parse_args()

    print("format", args.format)
    print("hidden dim", args.dim)
    print("output", args.output)
    print("graph-range", args.graph_range)
    if args.store_result:
        print("will store the pref result")

    layer, preprocess_func = load_layer_prepfunc(args)

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

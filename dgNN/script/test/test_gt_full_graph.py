import argparse
import pdb

import torch

from dgNN.layers import GTlayer, SparseMHA, SparseMHA_hyper_inference_timing
from dgNN.utils import load_data_full_graph, preprocess_CSR, preprocess_Hyper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT_full_graph")
    parser.add_argument("--format", type=str, default="csr")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    print("format:", args.format)
    print("dataset:", args.dataset)
    print("hidden dim", args.dim)
    print("num heads", args.heads)

    if args.format == "csr":
        layer = SparseMHA
        preprocess_func = preprocess_CSR
    elif args.format == "hyper":
        layer = SparseMHA_hyper_inference_timing
        preprocess_func = preprocess_Hyper
    else:
        raise ValueError(f"Unsupported format {args.format}")

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = load_data_full_graph(args.dataset, args.data_dir)
    g = dataset[0].to(dev)
    # Create the sparse adjacency matrix A.
    params = preprocess_func(g)
    params = [param.to(dev) for param in params]
    X = g.ndata["feat"]
    in_size = X.shape[1]
    GTlayer = GTlayer(
        layer, in_size=in_size, hidden_size=args.dim, num_heads=args.heads
    ).to(dev)

    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 2
    for epoch in range(50):
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

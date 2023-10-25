import argparse
import os

import pdb
import pickle
import sys

import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../.."))


def mean(arr):
    return sum(arr) / len(arr)


def main(args):
    formats = ["outdegree", "csr", "hyper", "hyper_nofuse"]
    avg_degrees = [2, 4, 8, 16, 32, 48, 64, 96, 128, 160]
    # avg_degrees = [2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112,128, 160]

    avg_degree_all = []
    time_no_fuse_all = []
    time_fuse_all = []

    for i, format in enumerate(formats):
        a1 = []
        a2 = []
        a3 = []
        for avg_d in avg_degrees:
            output = os.path.join(
                args.output, f"avg_d{avg_d}", f"{format}_{args.dim}_result.pkl"
            )
            with open(output, "rb") as f:
                avg_degree, time_no_fuse, time_fuse = pickle.load(f)
            a1.append(mean(avg_degree))
            a2.append(mean(time_no_fuse))
            a3.append(mean(time_fuse))
        avg_degree_all.append(a1)
        time_no_fuse_all.append(a2)
        time_fuse_all.append(a3)

    with open(
        os.path.join(args.output, f"avg_d{avg_degrees[0]}", f"0_config.pkl"), "rb"
    ) as f:
        config = pickle.load(f)
    print(config)
    save_dir = f"""/workspace2/fuse_attention/figure/graphworld/nvertex{config["nvertex"]}_pow{config["power_exponent"]}_dim{args.dim}"""
    print(save_dir)
    title = f"""nvertex={config["nvertex"]}, power_exponent={config["power_exponent"]}, dim={args.dim}"""
    fig = plt.figure(dpi=100, figsize=[12, 8])
    plt.ylabel("Elapsed time", fontsize=20)
    plt.xlabel("Average degree", fontsize=20)
    for i, format in enumerate(formats):
        plt.plot(avg_degree_all[0], time_fuse_all[i], "o-", label=format)
    plt.plot(avg_degree_all[0], time_no_fuse_all[0], "o-", label="Benchmark")

    # plt.xticks(avg_degrees)
    plt.title(title, fontsize=20)
    plt.legend()
    plt.savefig(save_dir + "_time.png")

    fig = plt.figure(dpi=100, figsize=[12, 8])
    plt.ylabel("Speedup", fontsize=20)
    plt.xlabel("Average degree", fontsize=20)
    for i, format in enumerate(formats):
        speedup_mean = np.array(time_no_fuse_all[0]) / np.array(time_fuse_all[i])
        plt.plot(avg_degree_all[0], speedup_mean, "o-", label=format)

    plt.title(title, fontsize=20)
    plt.legend()
    plt.savefig(save_dir + "_speedup.png")


if __name__ == "__main__":
    # Load graph data
    # parse argument
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--output", type=str)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")

    args = parser.parse_args()

    main(args)

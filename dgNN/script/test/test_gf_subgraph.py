import argparse
import pdb

import torch

from data.data import LoadData
from dgl.data import (
    CiteseerGraphDataset,
    CLUSTERDataset,
    CoraGraphDataset,
    PATTERNDataset,
    PubmedGraphDataset,
    Subset,
)
from dgl.dataloading import GraphDataLoader

from dgNN.layers import choose_GTlayer, SparseMHA_subgraph
from dgNN.utils import preprocess_SubGraph, train, train_SBM
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset
from ogb.lsc import DglPCQM4Mv2Dataset
from ogb.nodeproppred import DglNodePropPredDataset
from torch.utils.data import DataLoader


def load_dataset(dataset_name, data_dir):
    train_fn = train

    if dataset_name == "PCQM4Mv2-full" or dataset_name == "ogbg-molhiv":
        if dataset_name == "PCQM4Mv2-full":
            dataset = DglPCQM4Mv2Dataset(root=data_dir)
        else:
            dataset = DglGraphPropPredDataset(dataset_name, data_dir)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        dataset = dataset[train_idx]
    elif dataset_name == "MNIST" or dataset_name == "CIFAR10":
        dataset = LoadData(dataset_name)
    elif dataset_name == "PATTERN":
        train_fn = train_SBM
        dataset = PATTERNDataset(mode="train", raw_dir=data_dir)
    elif dataset_name == "CLUSTER":
        train_fn = train_SBM
        dataset = CLUSTERDataset(mode="train", raw_dir=data_dir)
    else:
        raise ValueError(f"unknown dataset {dataset_name}")
    return dataset, train_fn


def cal_available_node(dim, MAX_LIMIT=64 * 1024 / 4):
    MAX_NEIGH = 192
    return (MAX_LIMIT - MAX_NEIGH * 32) / (dim * 2)


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="/workspace2/dataset")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")

    args = parser.parse_args()
    print("Dataset", args.dataset)
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)

    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset, train_fn = load_dataset(args.dataset, args.data_dir)
    if args.dataset == "CLUSTER" or args.dataset == "PATTERN":
        num_nodes = torch.tensor([subgraph.num_nodes() for (subgraph) in dataset])
    elif args.dataset == "MNIST" or args.dataset == "CIFAR10":
        num_nodes = torch.tensor(
            [subgraph.num_nodes() for (subgraph, _) in dataset.train]
        )
    else:
        num_nodes = torch.tensor([subgraph.num_nodes() for (subgraph, _) in dataset])

    GTlayer = choose_GTlayer(
        args.dataset,
        MHAlayer=SparseMHA_subgraph,
        hidden_size=args.dim,
        num_heads=args.heads,
    )
    GTlayer = GTlayer.to(dev)
    max_nodes = cal_available_node(args.dim / args.heads)
    subgraph_index = torch.nonzero(num_nodes < max_nodes).squeeze().long()
    dataset_name = args.dataset
    if dataset_name == "MNIST" or dataset_name == "CIFAR10":
        collate_fn = dataset.collate
        dataset = Subset(dataset.train, subgraph_index.cpu())
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        dataset = dataset[subgraph_index]
        train_dataloader = GraphDataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )

    print(subgraph_index.squeeze(), subgraph_index.dtype)
    print("max nodes support in subgraph", max_nodes)
    print("num of satisfy subgraphs", subgraph_index.shape[0])

    print("GTlayer", GTlayer)
    time_no_fuse, time_fuse = train_fn(
        preprocess_SubGraph, GTlayer, train_dataloader, dev
    )

    # print("----------------------Forward------------------------")
    # time_no_fuse = []
    # time_fuse = []
    # warmup = 2
    # for i, (batched_g, labels) in enumerate(train_dataloader):
    #     # print("----------------------without fuse--------------------------")
    #     params = preprocess_SubGraph(batched_g)
    #     if params == None:
    #         continue
    #     params = [param.to(dev) for param in params]
    #     batched_g, labels = batched_g.to(dev), labels.to(dev)
    #     logits, elapsed_time = GTlayer(params, batched_g.ndata["feat"])
    #     print(f"epoch {i} non-fused time %.4f" % elapsed_time)
    #     if i > warmup:
    #         time_no_fuse.append(elapsed_time)
    #         # print("----------------------with fuse--------------------------")
    #         logits_fuse, elapsed_time = GTlayer(
    #             params, batched_g.ndata["feat"], fuse=True
    #         )
    #         time_fuse.append(elapsed_time)
    #         # pdb.set_trace()
    #         print(f"epoch {i} fused time %.4f" % elapsed_time)
    #         # if i < 5:
    #         #     check_correct(logits, logits_fuse, params)
    #         if i == 30:
    #             break

    print("----------------------Result------------------------")
    print(
        "no-fuse average time {:.4f} ms".format(
            sum(time_no_fuse[:-1]) / (len(time_no_fuse) - 1)
        )
    )
    print(
        "fuse average time {:.4f} ms".format(sum(time_fuse[:-1]) / (len(time_fuse) - 1))
    )

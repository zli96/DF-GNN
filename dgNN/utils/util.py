import os, pdb

from timeit import default_timer

import matplotlib.pyplot as plt
import ScheduleProfiler
import torch

import yaml
from data import LoadData
from dgl.data import (
    CiteseerGraphDataset,
    CLUSTERDataset,
    CoraGraphDataset,
    PATTERNDataset,
    PubmedGraphDataset,
)

from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset
from ogb.lsc import DglPCQM4Mv2Dataset
from ogb.nodeproppred import DglNodePropPredDataset

from torch.utils.data import DataLoader

profiler = ScheduleProfiler.ScheduleProfiler()

WARP_SIZE = 32


def load_dataset_fn(dataset_name, data_dir):
    train_fn = train
    collate_fn = None
    # train function for node-classification task
    if dataset_name in ["PascalVOC-SP", "COCO-SP", "PATTERN", "CLUSTER"]:
        train_fn = train_SBM

    if dataset_name in ["PCQM4Mv2-full", "ogbg-molhiv"]:
        if dataset_name == "PCQM4Mv2-full":
            dataset = DglPCQM4Mv2Dataset(root=data_dir)
        else:
            dataset = DglGraphPropPredDataset(dataset_name, data_dir)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        dataset = dataset[train_idx]
    elif dataset_name in ["MNIST", "CIFAR10"]:
        dataset_all = LoadData(dataset_name, data_dir)
        dataset = dataset_all.train
        collate_fn = dataset_all.collate
        # # TODO collate work?
        # train_dataloader = GraphDataLoader(
        #     trainset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate
        # )
    elif dataset_name in ["Peptides-func", "Peptides-struct"]:
        dataset = LoadData(dataset_name, data_dir)

    if dataset_name in ["PascalVOC-SP", "COCO-SP"]:
        dataset = LoadData(dataset_name, data_dir)
    elif dataset_name == "PATTERN":
        dataset = PATTERNDataset(mode="train", raw_dir=data_dir)
    elif dataset_name == "CLUSTER":
        dataset = CLUSTERDataset(mode="train", raw_dir=data_dir)

    if dataset == None:
        raise ValueError(f"unknown dataset {dataset_name}")

    return dataset, train_fn, collate_fn


def load_data_full_graph(dataset_name, dataset_dir):
    if dataset_name == "cora":
        dataset = CoraGraphDataset(dataset_dir)
    elif dataset_name == "arxiv":
        dataset = DglNodePropPredDataset("ogbn-arxiv")[0]
    elif dataset_name == "cite":
        dataset = CiteseerGraphDataset(dataset_dir)
    elif dataset_name == "pubmed":
        dataset = PubmedGraphDataset(dataset_dir)
    else:
        raise ValueError(f"Unsupport dataset {dataset_name}")
    return dataset


def figure_num_std(
    dataset_name, batch_size, num, num_avg, num_edges, num_edges_avg, std, std_avg
):
    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(num)
    plt.title(
        f"{dataset_name} # of nodes per batch, mean {num_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_num_nodes_{batch_size}.png")
    fig.clear()

    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(num_edges)
    plt.title(
        f"{dataset_name} # of edges per batch, mean {num_edges_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_num_edges_{batch_size}.png")
    fig.clear()

    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.boxplot(std)
    plt.title(
        f"{dataset_name} std of nodes per batch, mean {std_avg:.2f}",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_std_nodes_{batch_size}.png")


def figure_num_neigh_dist(dataset_name, num_neigh):
    max_neigh = 0
    for i in range(len(num_neigh) - 1, 0, -1):
        if num_neigh[i] != 0:
            max_neigh = i
            break
    num_neigh = num_neigh[:max_neigh]
    fig = plt.figure(dpi=100, figsize=[8, 8])
    plt.bar(range(len(num_neigh)), num_neigh)
    plt.title(
        f"{dataset_name} # of neighbors distribution",
        fontdict={"fontsize": 20},
    )
    for x1, y1 in enumerate(num_neigh):
        plt.text(x1, y1 + 10, y1.item(), ha="center", fontsize=16)
    plt.savefig(f"figure/{dataset_name}_num_neigh_dist.png")
    fig.clear()


def figure_nodes_neigh(dataset_name, num_neigh_per_node):
    fig = plt.figure()
    plt.plot(num_neigh_per_node, color="red")
    plt.title(
        f"{dataset_name} # of neighbors per nodes",
        fontdict={"fontsize": 20},
    )
    plt.savefig(f"figure/{dataset_name}_nodes_neigh.png")
    fig.clear()


def check_correct(logits, logits_fuse, params):
    check_same = torch.tensor(
        [all(i) for i in torch.isclose(logits, logits_fuse, rtol=0.001)]
    )
    if all(check_same):
        print("the results are the same, success!!!!!!!!!!")
    else:
        false_flag = torch.argwhere(~check_same)
        row_ptr = params[1]
        col_ind = params[2]
        acc = 0

        for i in false_flag:
            if not check_same[i]:
                chech_same_ele = torch.isclose(logits[i], logits_fuse[i], rtol=0.001)[0]
                if sum(chech_same_ele).item() + 1 != chech_same_ele.numel():
                    print(f"error node {i} mismatch")
                    print("neighbor nodes", col_ind[row_ptr[i] : row_ptr[i + 1]])
                    print("nonfuse result", logits[i])
                    print("fuse result", logits_fuse[i])
                    print(chech_same_ele)
                    acc = acc + 1
                if acc > 5:
                    break
        if acc == 0:
            print("the results are the same, success!!!!!!!!!!")


def Move2Device(data_list, dev):
    ### move data in list to dev
    data_dev = []
    for data in data_list:
        if isinstance(data, tuple):
            data_dev.append(
                [param.to(dev) if hasattr(param, "to") else param for param in data]
            )
        elif hasattr(data, "to"):
            data_dev.append(data.to(dev))
        else:
            data_dev.append(data)
    return data_dev


def train(process_func, layer, train_dataloader, dev, **kwargs):
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 1
    sample_start_time = 0
    for i, (batched_g, labels) in enumerate(train_dataloader):
        print(f"-----epoch {i}--------")
        print(
            f"epoch {i} sample elapsed time {default_timer() - sample_start_time:.2f} s"
        )
        ## preprocess
        params = process_func(batched_g, **kwargs)
        batched_g, labels, params = Move2Device([batched_g, labels, params], dev)
        ## run by DGL sparse API
        logits, elapsed_time = layer(params, batched_g.ndata["feat"])
        print(f"epoch {i} non-fused time %.4f" % elapsed_time)

        if i >= warmup:
            time_no_fuse.append(elapsed_time)
            ## run by fuse attention
            logits_fuse, elapsed_time = layer(
                params, batched_g.ndata["feat"], fuse=True
            )
            time_fuse.append(elapsed_time)
            print(f"epoch {i} fused time %.4f" % elapsed_time)
            if i < 2:
                check_correct(logits[:1000], logits_fuse[:1000], params)
            if i == 20:
                break
        sample_start_time = default_timer()
    return time_no_fuse, time_fuse


def train_SBM(process_func, layer, train_dataloader, dev, **kwargs):
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 1
    sample_start_time = 0
    for i, (batched_g) in enumerate(train_dataloader):
        print(f"-----epoch {i}--------")
        print(
            f"epoch {i} sample elapsed time {default_timer() - sample_start_time:.2f} s"
        )
        ## preprocess
        params = process_func(batched_g, **kwargs)
        batched_g, params = Move2Device([batched_g, params], dev)

        ## run by DGL sparse API
        logits, elapsed_time = layer(params, batched_g.ndata["feat"])
        print(f"epoch {i} non-fused time %.4f" % elapsed_time)
        if i >= warmup:
            time_no_fuse.append(elapsed_time)
            ## run by fuse attention
            logits_fuse, elapsed_time = layer(
                params, batched_g.ndata["feat"], fuse=True
            )
            time_fuse.append(elapsed_time)
            print(f"epoch {i} fused time %.4f" % elapsed_time)
            if i < 2:
                check_correct(logits[:1000], logits_fuse[:1000], params)
            if i == 20:
                break
        sample_start_time = default_timer()
    return time_no_fuse, time_fuse


def train_profile(process_func, layer, train_dataloader, dev, **arg):
    print("----------------------Forward------------------------")
    for i, (batched_g, labels) in enumerate(train_dataloader):
        # print("----------------------without fuse--------------------------")
        params = process_func(batched_g)
        params = [param.to(dev) for param in params]
        batched_g, labels = batched_g.to(dev), labels.to(dev)
        profiler.start()
        logits, elapsed_time = layer(params, batched_g.ndata["feat"], **arg)
        profiler.stop()
        # if i > warmup:
        #     time_no_fuse.append(elapsed_time)
        #     # print("----------------------with fuse--------------------------")
        #     logits_fuse, elapsed_time = layer(
        #         params, batched_g.ndata["feat"], fuse=True
        #     )
        #     time_fuse.append(elapsed_time)
        #     # pdb.set_trace()
        #     print(f"epoch {i} fused time %.4f" % elapsed_time)
        #     # if i < 5:
        #     #     check_correct(logits, logits_fuse, params)
    return


def train_profile_SBM(process_func, layer, train_dataloader, dev, **arg):
    print("----------------------Forward------------------------")
    for i, (batched_g) in enumerate(train_dataloader):
        # print("----------------------without fuse--------------------------")
        params = process_func(batched_g)
        params = [param.to(dev) for param in params]
        batched_g = batched_g.to(dev)
        profiler.start()
        logits, elapsed_time = layer(params, batched_g.ndata["feat"], **arg)
        profiler.stop()
        # if i > warmup:
        #     time_no_fuse.append(elapsed_time)
        #     # print("----------------------with fuse--------------------------")
        #     logits_fuse, elapsed_time = layer(
        #         params, batched_g.ndata["feat"], fuse=True
        #     )
        #     time_fuse.append(elapsed_time)
        #     # pdb.set_trace()
        #     print(f"epoch {i} fused time %.4f" % elapsed_time)
        #     # if i < 5:
        #     #     check_correct(logits, logits_fuse, params)
    return


class Timer:
    def __init__(self):
        self.timer = default_timer
        self.device = "cuda:0"

    def __enter__(self):
        if self.device == "cuda:0":
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()  # type: ignore
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if self.device == "cuda:0":
            self.end_event.record()  # type: ignore
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = self.start_event.elapsed_time(self.end_event) / 1e3
        else:
            self.elapsed_secs = self.timer() - self.tic


def benchmark(function, *args):
    # dry run
    for i in range(3):
        out = function(*args)

    with Timer() as t:
        for i in range(100):
            out = function(*args)

    return out, t.elapsed_secs / 100


def parse_args(parser):
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as file:
            data = yaml.safe_load(file)
        delattr(args, "config")
        arg_dict = args.__dict__
        for key, value in data.items():
            if key not in arg_dict.keys() or arg_dict[key] == None:
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
    return args


def parser_argument(parser):
    parser.add_argument("--config", type=str)
    parser.add_argument("--format", type=str, default="csr")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--store-result", action="store_true")
    parser.add_argument("--subgraph-filter", action="store_true")
    args = parse_args(parser)
    print("Dataset", args.dataset)
    print("format", args.format)
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    if args.subgraph_filter:
        print("will filter the subgraph bigger than limit")
    if args.store_result:
        print("will store the pref result")
    return args


def mkdir(path):

    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("Create new folder")
    else:
        print("folder existed")

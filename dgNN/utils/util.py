import os.path as osp
import pdb

import ssl
import sys
import urllib
from timeit import default_timer
from typing import Optional

import dgl.sparse as dglsp

import matplotlib.pyplot as plt
import ScheduleProfiler
import torch
from data import LoadData
from dgl.data import (
    CiteseerGraphDataset,
    CLUSTERDataset,
    CoraGraphDataset,
    PATTERNDataset,
    PubmedGraphDataset,
)
from dgl.data.utils import makedirs

from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset
from ogb.lsc import DglPCQM4Mv2Dataset
from ogb.nodeproppred import DglNodePropPredDataset

from torch.utils.data import DataLoader

profiler = ScheduleProfiler.ScheduleProfiler()


def download_url(
    url: str, folder: str, log: bool = True, filename: Optional[str] = None
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def load_data_batch(dataset_name, batch_size, data_dir):
    train_fn = train

    if dataset_name == "PCQM4Mv2-full" or dataset_name == "ogbg-molhiv":
        if dataset_name == "PCQM4Mv2-full":
            dataset = DglPCQM4Mv2Dataset(root=data_dir)
        else:
            dataset = DglGraphPropPredDataset(dataset_name, data_dir)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        train_dataloader = GraphDataLoader(
            dataset[train_idx],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_dgl,
        )
    elif dataset_name == "MNIST" or dataset_name == "CIFAR10":
        dataset = LoadData(dataset_name)
        trainset, _, _ = dataset.train, dataset.val, dataset.test
        train_dataloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate
        )
    elif dataset_name == "PATTERN":
        train_fn = train_SBM
        dataset = PATTERNDataset(mode="train", raw_dir=data_dir)
        train_dataloader = GraphDataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
    elif dataset_name == "CLUSTER":
        train_fn = train_SBM
        dataset = CLUSTERDataset(mode="train", raw_dir=data_dir)
        train_dataloader = GraphDataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
    else:
        raise ValueError(f"unknown dataset {dataset_name}")
    return train_dataloader, train_fn


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


def preprocess_CSR(g):
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = torch.tensor([A.val[i] for i in val_idx]).float()
    return A, row_ptr, col_ind, val


def preprocess_Hyper(g):
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    rows = A.row.int()
    rows = torch.sort(rows).values
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = torch.tensor([A.val[i] for i in val_idx]).float()
    return A, row_ptr, col_ind, rows, val


def preprocess_SubGraph(g):
    nodes = g.batch_num_nodes()
    # print("max num of nodes", max(nodes).item())
    nodes_subgraph = torch.cat(
        (torch.tensor([0]), torch.cumsum(nodes.clone(), 0))
    ).int()
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = torch.tensor([A.val[i] for i in val_idx]).float()
    return A, row_ptr, col_ind, nodes_subgraph, val


def preprocess_ELL(
    g,
    bucket_sizes=[],
    num_col_parts=1,
):
    raise ValueError("ELL is deprecated ")
    # indices = torch.stack(g.edges())
    # N = g.num_nodes()
    # A = dglsp.spmatrix(indices, shape=(N, N))
    # row_ptr, col_ind, val_idx = A.csr()

    # row_ptr = row_ptr.int()
    # col_ind = col_ind.int()
    # val = torch.tensor([A.val[i] for i in val_idx]).float()

    # # cluster the rows into diff buckets based on its num of neighbors
    # row_col_ind, _, _ = format_conversion.csr2ell(
    #     N, N, row_ptr, col_ind, num_col_parts, bucket_sizes
    # )

    # # num of elements each tb need to process
    # elements_per_tb = 4
    # rows_per_tb = []
    # row_col_ind = row_col_ind[0]

    # # calculate the num of elements each tb need to process
    # for i, bucket_size in enumerate(bucket_sizes):
    #     num_elements = len(row_col_ind[i])
    #     num_rows = elements_per_tb // bucket_size
    #     rows_per_tb = rows_per_tb + [num_rows] * (num_elements // num_rows)
    #     res = num_elements % num_rows
    #     if res != 0:
    #         rows_per_tb = rows_per_tb + [res]
    # row_index = torch.cat(row_col_ind, 0).int()
    # rows_per_tb = torch.cat(
    #     (torch.tensor([0]), torch.cumsum(torch.tensor(rows_per_tb), 0))
    # ).int()

    # return A, row_ptr, col_ind, row_index, rows_per_tb, val


def check_correct(logits, logits_fuse, params):
    if all(torch.isclose(logits, logits_fuse, atol=0.001).flatten()):
        print("the results are the same, success!!!!!!!!!!")
    else:
        if len(params) == 5:
            row_ptr = params[2]
            col_ind = params[3]
        else:
            row_ptr = params[1]
            col_ind = params[2]
        for i in range(logits.shape[0]):
            if not all(torch.isclose(logits[i], logits_fuse[i], atol=0.001).flatten()):
                print(f"error node {i} mismatch")
                print("neighbor nodes", col_ind[row_ptr[i] : row_ptr[i + 1]])
                print(logits[i])
                print(logits_fuse[i])
                pdb.set_trace()
            else:
                print("----------------pass------------------")
                print("neighbor nodes", col_ind[row_ptr[i] : row_ptr[i + 1]])
                print("")


def train(process_func, layer, train_dataloader, dev, **arg):
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 1
    sample_start_time = 0
    for i, (batched_g, labels) in enumerate(train_dataloader):
        pdb.set_trace()
        print(
            f"epoch {i} sample elapsed time {default_timer() - sample_start_time:.2f} s"
        )
        params = process_func(batched_g, **arg)
        if params == None:
            continue
        params = [param.to(dev) for param in params]
        batched_g, labels = batched_g.to(dev), labels.to(dev)
        logits, elapsed_time = layer(params, batched_g.ndata["feat"])
        print(f"epoch {i} non-fused time %.4f" % elapsed_time)
        if i > warmup:
            time_no_fuse.append(elapsed_time)
            # print("----------------------with fuse--------------------------")
            logits_fuse, elapsed_time = layer(
                params, batched_g.ndata["feat"], fuse=True
            )
            time_fuse.append(elapsed_time)
            # pdb.set_trace()
            print(f"epoch {i} fused time %.4f" % elapsed_time)
            if i < 3:
                check_correct(logits, logits_fuse, params)
            if i == 20:
                break
        sample_start_time = default_timer()
    return time_no_fuse, time_fuse


def train_SBM(process_func, layer, train_dataloader, dev, **arg):
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 1
    sample_start_time = 0
    for i, (batched_g) in enumerate(train_dataloader):
        print(
            f"epoch {i} sample elapsed time {default_timer() - sample_start_time:.2f} s"
        )
        params = process_func(batched_g, **arg)
        if params == None:
            continue
        params = [param.to(dev) for param in params]
        batched_g = batched_g.to(dev)
        logits, elapsed_time = layer(params, batched_g.ndata["feat"])
        print(f"epoch {i} non-fused time %.4f" % elapsed_time)
        if i > warmup:
            time_no_fuse.append(elapsed_time)
            logits_fuse, elapsed_time = layer(
                params, batched_g.ndata["feat"], fuse=True
            )
            time_fuse.append(elapsed_time)
            # pdb.set_trace()
            print(f"epoch {i} fused time %.4f" % elapsed_time)
            if i < 3:
                check_correct(logits, logits_fuse, params)
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


def parser_argument(parser):
    parser.add_argument("--format", type=str, default="csr")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")

    args = parser.parse_args()
    print("Dataset", args.dataset)
    print("format: ", args.format)
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    return args

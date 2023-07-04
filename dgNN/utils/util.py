import dgl.sparse as dglsp
import format_conversion
import matplotlib.pyplot as plt
import numpy as np
import ScheduleProfiler
import torch
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset
from ogb.lsc import DglPCQM4Mv2Dataset
from ogb.nodeproppred import DglNodePropPredDataset

profiler = ScheduleProfiler.ScheduleProfiler()


def load_data_batch(dataset_name, batch_size):
    if dataset_name == "PCQM4Mv2-full":
        dataset = DglPCQM4Mv2Dataset(root="./data")
    else:
        dataset = DglGraphPropPredDataset(dataset_name, "./data/OGB")
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    train_dataloader = GraphDataLoader(
        dataset[train_idx], batch_size=batch_size, shuffle=False, collate_fn=collate_dgl
    )
    return train_dataloader


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


def preprocess_ELL(
    g,
    bucket_sizes=[],
    num_col_parts=1,
):
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    row_ptr, col_ind, val_idx = A.csr()

    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = torch.tensor([A.val[i] for i in val_idx]).float()

    # cluster the rows into diff buckets based on its num of neighbors
    row_col_ind, _, _ = format_conversion.csr2ell(
        N, N, row_ptr, col_ind, num_col_parts, bucket_sizes
    )

    # num of elements each tb need to process
    elements_per_tb = 4
    rows_per_tb = []
    row_col_ind = row_col_ind[0]

    # calculate the num of elements each tb need to process
    for i, bucket_size in enumerate(bucket_sizes):
        num_elements = len(row_col_ind[i])
        num_rows = elements_per_tb // bucket_size
        rows_per_tb = rows_per_tb + [num_rows] * (num_elements // num_rows)
        res = num_elements % num_rows
        if res != 0:
            rows_per_tb = rows_per_tb + [res]
    row_index = torch.cat(row_col_ind, 0).int()
    rows_per_tb = torch.cat(
        (torch.tensor([0]), torch.cumsum(torch.tensor(rows_per_tb), 0))
    ).int()

    return A, row_ptr, col_ind, row_index, rows_per_tb, val


def train(process_func, layer, train_dataloader, dev, **arg):
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 2
    for i, (batched_g, labels) in enumerate(train_dataloader):
        # print("----------------------without fuse--------------------------")
        params = process_func(batched_g, **arg)
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
            if i < 5:
                if all(torch.isclose(logits, logits_fuse, atol=0.001).flatten()):
                    print("the results are the same, success!!!!!!!!!!")
                else:
                    row_ptr = params[1]
                    col_ind = params[2]
                    for i in range(logits.shape[0]):
                        if not all(
                            torch.isclose(
                                logits[i], logits_fuse[i], atol=0.001
                            ).flatten()
                        ):
                            print(f"error node {i} mismatch")
                            print(
                                "neighbor nodes", col_ind[row_ptr[i] : row_ptr[i + 1]]
                            )
                            print(logits[i])
                            print(logits_fuse[i])
                        else:
                            print("----------------pass------------------")
                            print(
                                "neighbor nodes", col_ind[row_ptr[i] : row_ptr[i + 1]]
                            )
                            print("")
                    exit()
            if i == 30:
                break
    return time_no_fuse, time_fuse


def train_profile(process_func, layer, train_dataloader, dev, **arg):
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 5
    for i, (batched_g, labels) in enumerate(train_dataloader):
        # print("----------------------without fuse--------------------------")
        params = process_func(batched_g, **arg)
        params = [param.to(dev) for param in params]
        batched_g, labels = batched_g.to(dev), labels.to(dev)
        profiler.start()
        logits, elapsed_time = layer(params, batched_g.ndata["feat"])
        profiler.stop()
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
            if all(torch.isclose(logits, logits_fuse, atol=0.001).flatten()):
                print("the results are the same, success!!!!!!!!!!")
            else:
                for i in range(logits.shape[0]):
                    if not all(
                        torch.isclose(logits[i], logits_fuse[i], atol=0.001).flatten()
                    ):
                        print(f"error node {i} mismatch")
                        # print("neighbor nodes", col_ind[row_ptr[i]:row_ptr[i+1]])
                        print(logits[i])
                        print(logits_fuse[i])
            if i == 30:
                break
    return time_no_fuse, time_fuse

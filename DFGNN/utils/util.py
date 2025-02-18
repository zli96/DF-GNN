import os

from timeit import default_timer

import dgl.sparse as dglsp

import matplotlib.pyplot as plt

import torch

import yaml
# ZL: some of those are deprecated in the new dgl
# I don't use the dataloading functions anyways.
# from dgl.data import CoraGraphDataset, CoraFullDataset
# from dgl.data.lrgb import (
#     AsGraphPredDataset,
#     CIFAR10SuperPixelDataset,
#     CiteseerGraphDataset,
#     CLUSTERDataset,
#     COCOSuperpixelsDataset,
#     MNISTSuperPixelDataset,
#     PATTERNDataset,
#     PeptidesFunctionalDataset,
#     PeptidesStructuralDataset,
#     PubmedGraphDataset,
#     RedditDataset,
#     VOCSuperpixelsDataset,
# )

# from ogb.graphproppred import DglGraphPropPredDataset
# from ogb.linkproppred import DglLinkPropPredDataset
# from ogb.lsc import DglPCQM4Mv2Dataset
# from ogb.nodeproppred import DglNodePropPredDataset


datasets_NC = ["PascalVOC-SP", "COCO-SP", "PATTERN", "CLUSTER"]

WARP_SIZE = 32


# def LoadData(DATASET_NAME, data_dir):

#     # handling for MNIST or CIFAR Superpixels
#     if DATASET_NAME == "MNIST":
#         return MNISTSuperPixelDataset(data_dir)

#     if DATASET_NAME == "CIFAR10":
#         return CIFAR10SuperPixelDataset(data_dir)

#     # handling for LRGB datasets
#     LRGB_DATASETS = ["PascalVOC-SP", "COCO-SP", "Peptides-func", "Peptides-struct"]
#     if DATASET_NAME in LRGB_DATASETS:
#         if DATASET_NAME == "PascalVOC-SP":
#             return VOCSuperpixelsDataset(data_dir)
#         elif DATASET_NAME == "COCO-SP":
#             return COCOSuperpixelsDataset(data_dir)
#         elif DATASET_NAME == "Peptides-func":
#             return PeptidesFunctionalDataset(data_dir)
#         elif DATASET_NAME == "Peptides-struct":
#             return PeptidesStructuralDataset(data_dir)

#     raise ValueError("Unknown dataset: {}".format(DATASET_NAME))


# def load_dataset_fn(dataset_name, data_dir):
#     train_fn = inference_Graph_level
#     if dataset_name in datasets_NC:
#         train_fn = inference_Node_level

#     if dataset_name in ["PCQM4Mv2-full", "ogbg-molhiv"]:
#         if dataset_name == "PCQM4Mv2-full":
#             dataset = DglPCQM4Mv2Dataset(root=data_dir)
#         else:
#             dataset = AsGraphPredDataset(
#                 DglGraphPropPredDataset(dataset_name, data_dir)
#             )
#     elif dataset_name in [
#         "MNIST",
#         "CIFAR10",
#         "Peptides-func",
#         "Peptides-struct",
#         "PascalVOC-SP",
#         "COCO-SP",
#     ]:
#         dataset = LoadData(dataset_name, data_dir)
#     elif dataset_name == "PATTERN":
#         dataset = PATTERNDataset(mode="train", raw_dir=data_dir)
#     elif dataset_name == "CLUSTER":
#         dataset = CLUSTERDataset(mode="train", raw_dir=data_dir)
#     else:
#         raise ValueError(f"unknown dataset {dataset_name}")
#     return dataset, train_fn


# def preprocess_proteins(graph):
#     import dgl.function as fn

#     graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))
#     graph.create_formats_()

#     return graph


# def load_data_full_graph(dataset_name, dataset_dir):
#     if dataset_name == "cora":
#         dataset = CoraGraphDataset(raw_dir=dataset_dir)
#     elif dataset_name == "cora-full":
#         dataset = CoraFullDataset(raw_dir=dataset_dir)
#     elif dataset_name == "arxiv":
#         dataset = DglNodePropPredDataset(name="ogbn-arxiv", root=dataset_dir)
#         dataset = dataset[0]
#     elif dataset_name == "protein":
#         dataset = DglNodePropPredDataset(name="ogbn-proteins", root=dataset_dir)
#         g, _ = dataset[0]
#         g = preprocess_proteins(g)
#         dataset = [g]
#     elif dataset_name == "product":
#         dataset = DglNodePropPredDataset(name="ogbn-products", root=dataset_dir)
#         g, _ = dataset[0]
#         dataset = [g]
#     elif dataset_name == "ppa":
#         dataset = DglLinkPropPredDataset(name="ogbl-ppa", root=dataset_dir)
#         g = dataset[0]
#         g.ndata["feat"] = g.ndata["feat"].float()
#         dataset = [g]
#     elif dataset_name == "collab":
#         dataset = DglLinkPropPredDataset(name="ogbl-collab", root=dataset_dir)
#     elif dataset_name == "cite":
#         dataset = CiteseerGraphDataset(raw_dir=dataset_dir)
#     elif dataset_name == "pubmed":
#         dataset = PubmedGraphDataset(raw_dir=dataset_dir)
#     elif dataset_name == "reddit":
#         dataset = RedditDataset(raw_dir=dataset_dir)
#     elif dataset_name == "yelp":
#         dataset = dgl.data.YelpDataset(raw_dir=dataset_dir)
#     elif dataset_name == "Flickr":
#         dataset = dgl.data.FlickrDataset(raw_dir=dataset_dir)
#     elif dataset_name == "AmazonCoBuyComputer":
#         dataset = dgl.data.AmazonCoBuyComputerDataset(raw_dir=dataset_dir)
#     elif dataset_name == "AmazonCoBuyPhoto":
#         dataset = dgl.data.AmazonCoBuyPhotoDataset(raw_dir=dataset_dir)
#     elif dataset_name == "CoauthorCS":
#         dataset = dgl.data.CoauthorCSDataset(raw_dir=dataset_dir)
#     elif dataset_name == "CoauthorPhysics":
#         dataset = dgl.data.CoauthorPhysicsDataset(raw_dir=dataset_dir)
#     else:
#         raise ValueError(f"Unsupport dataset {dataset_name}")
#     return dataset


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
        # row_ptr = params[1]
        # col_ind = params[2]
        acc = 0

        for i in false_flag:
            if not check_same[i]:
                chech_same_ele = torch.isclose(logits[i], logits_fuse[i], rtol=0.001)[0]
                if sum(chech_same_ele).item() + 1 != chech_same_ele.numel():
                    print(f"error node {i} mismatch")
                    # print("neighbor nodes", col_ind[row_ptr[i] : row_ptr[i + 1]])
                    print("nonfuse result", logits[i])
                    print("fuse result", logits_fuse[i])
                    print(chech_same_ele)
                    acc = acc + 1
                if acc > 0:
                    break
        if acc == 0:
            print("the results are the same, success!!!!!!!!!!")


def preprocess_dglsp(g, **args):
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    return A


def inference_Graph_level(process_func, model, train_dataloader, dev):
    r"""training function for the graph-level task"""
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
        batched_g, labels = batched_g.to(dev), labels.to(dev)
        params = preprocess_dglsp(batched_g)

        ## run by DGL sparse API
        model.eval()
        logits, elapsed_time = model(params, batched_g.ndata["feat"])
        print(f"epoch {i} non-fused time %.4f" % elapsed_time)

        if i >= warmup:
            time_no_fuse.append(elapsed_time)
            ## run by fuse attention
            model.eval()
            params = process_func(batched_g)
            logits_fuse, elapsed_time = model(
                params, batched_g.ndata["feat"], fuse=True
            )
            time_fuse.append(elapsed_time)
            print(f"epoch {i} fused time %.4f" % elapsed_time)
            if i < 3:
                check_correct(logits[:1000], logits_fuse[:1000], params)
                check_correct(logits[-1000:], logits_fuse[-1000:], params)
            if i == 20:
                break
        sample_start_time = default_timer()
    return time_no_fuse, time_fuse


def inference_Node_level(process_func, model, train_dataloader, dev):
    r"""training function for the node-level task"""
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
        batched_g = batched_g.to(dev)
        params = preprocess_dglsp(batched_g)

        ## run by DGL sparse API
        model.eval()
        logits, elapsed_time = model(params, batched_g.ndata["feat"])
        print(f"epoch {i} non-fused time %.4f" % elapsed_time)
        if i >= warmup:
            time_no_fuse.append(elapsed_time)
            ## run by fuse attention
            model.eval()
            params = process_func(batched_g)
            logits_fuse, elapsed_time = model(
                params, batched_g.ndata["feat"], fuse=True
            )
            time_fuse.append(elapsed_time)
            print(f"epoch {i} fused time %.4f" % elapsed_time)
            if i < 3:
                check_correct(logits[:1000], logits_fuse[:1000], params)
                check_correct(logits[-1000:], logits_fuse[-1000:], params)
            if i == 20:
                break
        sample_start_time = default_timer()
    return time_no_fuse, time_fuse


def train_profile(process_func, model, train_dataloader, dev, args):
    import ScheduleProfiler

    profiler = ScheduleProfiler.ScheduleProfiler()
    fuse_flag = args.format != "nofuse"
    print("----------------------Forward------------------------")
    if args.dataset in datasets_NC:
        for i, (batched_g) in enumerate(train_dataloader):
            batched_g = batched_g.to(dev)
            params = process_func(batched_g)
            profiler.start()
            logits, elapsed_time = model(
                params, batched_g.ndata["feat"], fuse=fuse_flag
            )
            profiler.stop()
    else:
        for i, (batched_g, labels) in enumerate(train_dataloader):
            batched_g, labels = batched_g.to(dev), labels.to(dev)
            params = process_func(batched_g)
            profiler.start()
            logits, elapsed_time = model(
                params, batched_g.ndata["feat"], fuse=fuse_flag
            )
            profiler.stop()
    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(wait=5, warmup=1, active=1, repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         f"./log/tensor_board/{args.dataset}_{args.conv}_{args.format}"
    #     ),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    # ) as prof:
    #     for i, (batched_g) in enumerate(train_dataloader):
    #         batched_g = batched_g.to(dev)
    #         params = process_func(batched_g, **arg)
    #         logits, elapsed_time = model(
    #             params, batched_g.ndata["feat"], fuse=args.format != "nofuse"
    #         )
    #         prof.step()
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
        for i in range(10):
            out = function(*args)

    return out, t.elapsed_secs / 10


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
    parser.add_argument("--conv", type=str, default="gt")
    parser.add_argument("--format", type=str, default="all")
    parser.add_argument("--dim", type=int)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--data-dir", type=str, default="./data/OGB")
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--store-result", action="store_true")
    parser.add_argument("--subgraph-filter", action="store_true")
    parser.add_argument("--profile", action="store_true")

    args = parse_args(parser)
    print("GraphConv", args.conv)
    print("Dataset", args.dataset)
    print("format", args.format)
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    if args.subgraph_filter:
        print("will filter the subgraph bigger than limit")
    if args.store_result:
        print("will store the pref result")
    if args.profile:
        print("----------enter the profile mode-------------")
    return args


def mkdir(path):

    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("Create new folder")
    else:
        print("folder existed")

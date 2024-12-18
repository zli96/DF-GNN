import warnings
from functools import partial

import dgl
import dgl.sparse as dglsp

import pylibcugraphops.pytorch as ops_torch
import torch

from .AGNN import (
    AGNNConv_csr,
    AGNNConv_csr_gm,
    AGNNConv_cugraph,
    AGNNConv_hyper,
    AGNNConv_pyg,
    AGNNConv_softmax,
    AGNNConv_softmax_gm,
    AGNNConv_tiling,
)

from .GAT import (
    GATConv_cugraph,
    GATConv_dgNN,
    GATConv_hybrid,
    GATConv_hyper,
    GATConv_hyper_ablation,
    GATConv_hyper_recompute,
    GATConv_hyper_v2,
    GATConv_pyg,
    GATConv_softmax,
    GATConv_softmax_gm,
    GATConv_tiling,
)

from .GT import (
    SparseMHA_CSR,
    SparseMHA_CSR_GM,
    SparseMHA_cugraph,
    SparseMHA_forward_timing,
    SparseMHA_hybrid,
    SparseMHA_hyper,
    SparseMHA_hyper_ablation,
    SparseMHA_pyg,
    SparseMHA_softmax,
    SparseMHA_softmax_gm,
    SparseMHA_tiling,
)

WARP_SIZE = 32


def g_to_SPmatrix(g):
    indices = torch.stack(g.edges())
    # max_neigh = max(torch.bincount(indices[0]))
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    return A, 128


def preprocess_pyg(g, **args):
    src, dst = g.edges()
    edge_index = torch.stack((src, dst))
    return edge_index


def preprocess_CSR(g, **args):
    A, max_neigh = g_to_SPmatrix(g)

    # using max_degree to cal max smem consume
    # max_degree = int(max(A.sum(1)).item())
    smem_consume = (max_neigh + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    print("preprocess smem consume", smem_consume)

    # the CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    return row_ptr, col_ind, val, smem_consume


def preprocess_Hyper(g, **args):
    A, max_neigh = g_to_SPmatrix(g)

    # using max_degree to cal max smem consume
    # max_degree = int(max(A.sum(1)).item())
    # print("max degree of all nodes:",max_degree)
    smem_consume = (max_neigh * 8 + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    print("preprocess smem consume", smem_consume)

    # A.row: the src node of each edge
    rows = A.row.int()
    rows = torch.sort(rows).values

    # the CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    return row_ptr, col_ind, rows, val, smem_consume


def preprocess_cugraph(g, is_bipartite=False):
    max_in_degree = -1
    offsets, indices, _ = g.adj_tensors("csr")
    graph = ops_torch.CSC(
        offsets=offsets,
        indices=indices,
        num_src_nodes=g.num_src_nodes(),
        dst_max_in_degree=max_in_degree,
        is_bipartite=is_bipartite,
    )
    return graph


def preprocess_Hyper_fw_bw(g, fused=True):
    # print("start preprocess")
    A, max_neigh = g_to_SPmatrix(g)
    if not fused:
        return A, None, None, None, None, None, None, None, None

    # using max_degree to cal max smem consume
    # max_degree = int(max(A.sum(1)).item())
    # print(max_degree)
    smem_consume = (max_neigh * 8 + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    # print("preprocess smem consume", smem_consume)
    # A.row: the src node of each edge
    rows = A.row.int()
    rows = torch.sort(rows).values

    # the CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    A_csr = dglsp.from_csr(indptr=row_ptr, indices=col_ind, val=val)

    # the CSC format of adj matrix
    col_ptr, row_ind, val_idx = A_csr.csc()
    col_ptr = col_ptr.int()
    row_ind = row_ind.int()
    return A, rows, row_ptr, col_ind, val, col_ptr, row_ind, val_idx, smem_consume


def preprocess_softmax(g, **args):
    A, max_neigh = g_to_SPmatrix(g)

    # using max_degree to cal max smem consume
    # max_degree = int(max(A.sum(1)).item())
    smem_consume = (max_neigh + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    print("preprocess smem consume", smem_consume)

    # A.row: the src node of each edge
    rows = A.row.int()
    rows = torch.sort(rows).values

    # the CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    return row_ptr, col_ind, rows, val, smem_consume


def preprocess_hybrid(g):
    print("preprocess hybrid")
    g = dgl.add_self_loop(g)
    indices = torch.stack(g.edges())
    # max_neigh = max(torch.bincount(indices[0]))
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    val = torch.ones([g.num_edges(), 1], device=g.device)
    A2 = dglsp.val_like(A, val=val)
    return A, A2


def preprocess_SubGraph(g):
    nodes = g.batch_num_nodes()

    # num of nodes in each sub-graph(accumulate)
    nodes_subgraph = torch.cat(
        (torch.tensor([0]), torch.cumsum(nodes.clone(), 0))
    ).int()
    A, max_neigh = g_to_SPmatrix(g)

    # the CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    return A, row_ptr, col_ind, nodes_subgraph, val


def cal_available_node(dim, MAX_NEIGH, MAX_LIMIT=64 * 1024 / 4):
    block_size = 1024 / dim
    ## smem need to store
    ## K,V features: 2*num_store_nodes*dim
    ## warpLevelSums: blocksize * WARP_SIZE
    ## SDDMM result: MAX_NEIGH * blocksize
    warpLevelSums_overhead = WARP_SIZE * block_size
    neigh_nodes_weight_overhead = (
        (MAX_NEIGH + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE * block_size
    )
    return int(
        (MAX_LIMIT - warpLevelSums_overhead - neigh_nodes_weight_overhead) / (dim * 2)
    )


def preprocess_indegree_hyper(g, dim):
    ## global graph config
    # nodes_subgraph: num of nodes in each sub-graph(accumulate), shape(num_subgraph+1)
    nodes = g.batch_num_nodes()
    nodes_subgraph = torch.cat(
        (torch.tensor([0]), torch.cumsum(nodes.clone(), 0))
    ).int()
    A, max_neigh = g_to_SPmatrix(g)

    # A.row: the src node of each edge
    rows = A.row.int()
    rows = torch.sort(rows).values

    N = A.shape[0]
    in_degree = A.sum(0).int()
    num_neighbor = A.sum(1).int()
    if any(num_neighbor == 0):
        warnings.warn("exit zero-degree node")
    max_neighbor = max(num_neighbor).item()
    max_nodes = cal_available_node(dim, max_neighbor)
    print("max neighbor", max_neighbor)
    print("max supported num of nodes", max_nodes)

    ## store_node: the ind ex of nodes stored in smem, shape(num_subgraph*max_nodes,1)
    ## store_flag: the flag whether stored in smem, shape(N, 1)
    ## If: -1, not in smem
    ## If: int, the local index in smem
    ## smem_nodes_subgraph: num of nodes in smem of each subgraph, shape(num_subgraph+1)
    store_node = []
    store_flag = torch.zeros(N, dtype=torch.int) - 1
    smem_nodes_subgraph = [0]
    cumsum = 0
    ## Loop over all subgraph
    for i in range(g.batch_size):
        node_lb = nodes_subgraph[i]
        node_hb = nodes_subgraph[i + 1]
        in_degree_local = in_degree[node_lb:node_hb]
        degree_limit = sum(in_degree_local > 1).item()
        max_nodes_local = min(max_nodes, degree_limit)
        cumsum += max_nodes_local
        smem_nodes_subgraph.append(cumsum)
        _, indices = torch.sort(in_degree_local, descending=True)
        store_node_subgraph = indices[:max_nodes_local] + node_lb
        store_node += store_node_subgraph.tolist()
        store_flag[store_node_subgraph] = torch.arange(0, max_nodes_local).int()

    smem_nodes_subgraph = torch.tensor(smem_nodes_subgraph).int()
    store_node = torch.tensor(store_node).int()

    ## The CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    assert g.batch_size + 1 == nodes_subgraph.shape[0]
    assert g.batch_size + 1 == smem_nodes_subgraph.shape[0]
    print(
        f"{smem_nodes_subgraph[-1].item()} nodes of all {nodes_subgraph[-1].item()} nodes are stored in smem"
    )
    return (
        A,
        rows,
        row_ptr,
        col_ind,
        val,
        nodes_subgraph,
        smem_nodes_subgraph,
        store_node,
        store_flag,
    )


def preprocess_indegree(g, dim):
    ## global graph config
    # nodes_subgraph: num of nodes in each sub-graph(accumulate), shape(num_subgraph+1)
    nodes = g.batch_num_nodes()
    nodes_subgraph = torch.cat(
        (torch.tensor([0]), torch.cumsum(nodes.clone(), 0))
    ).int()
    A, max_neigh = g_to_SPmatrix(g)
    N = A.shape[0]
    in_degree = A.sum(0).int()
    num_neighbor = A.sum(1).int()
    if any(num_neighbor == 0):
        warnings.warn("exit zero-degree node")
    max_neighbor = max(num_neighbor).item()
    max_nodes = cal_available_node(dim, max_neighbor)
    print("max neighbor", max_neighbor)
    print("max supported num of nodes", max_nodes)

    ## store_node: the ind ex of nodes stored in smem, shape(num_subgraph*max_nodes,1)
    ## store_flag: the flag whether stored in smem, shape(N, 1)
    ## If: -1, not in smem
    ## If: int, the local index in smem
    ## smem_nodes_subgraph: num of nodes in smem of each subgraph, shape(num_subgraph+1)
    store_node = []
    store_flag = torch.zeros(N, dtype=torch.int) - 1
    smem_nodes_subgraph = [0]
    cumsum = 0
    ## Loop over all subgraph
    for i in range(g.batch_size):
        node_lb = nodes_subgraph[i]
        node_hb = nodes_subgraph[i + 1]
        in_degree_local = in_degree[node_lb:node_hb]
        degree_limit = sum(in_degree_local > 1).item()
        max_nodes_local = min(max_nodes, degree_limit)
        cumsum += max_nodes_local
        smem_nodes_subgraph.append(cumsum)
        _, indices = torch.sort(in_degree_local, descending=True)
        store_node_subgraph = indices[:max_nodes_local] + node_lb
        store_node += store_node_subgraph.tolist()
        store_flag[store_node_subgraph] = torch.arange(0, max_nodes_local).int()

    smem_nodes_subgraph = torch.tensor(smem_nodes_subgraph).int()
    store_node = torch.tensor(store_node).int()

    ## The CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    assert g.batch_size + 1 == nodes_subgraph.shape[0]
    assert g.batch_size + 1 == smem_nodes_subgraph.shape[0]
    print(
        f"{smem_nodes_subgraph[-1].item()} nodes of all {nodes_subgraph[-1].item()} nodes are stored in smem"
    )
    return (
        A,
        row_ptr,
        col_ind,
        val,
        nodes_subgraph,
        smem_nodes_subgraph,
        store_node,
        store_flag,
    )


def subgraph_filter(dataset, dataset_name, dim, heads):
    if dataset_name in ["PascalVOC-SP", "COCO-SP", "PATTERN", "CLUSTER"]:
        num_nodes = torch.tensor([subgraph.num_nodes() for (subgraph) in dataset])
    else:
        num_nodes = torch.tensor([subgraph.num_nodes() for (subgraph, _) in dataset])
    # max_neighbor = max(A.sum(1).int()).item()
    max_nodes = cal_available_node(dim / heads, 192)
    subgraph_index = torch.nonzero(num_nodes < max_nodes).squeeze().long()
    if dataset_name in ["MNIST", "CIFAR10"]:
        dataset = Subset(dataset, subgraph_index.cpu())
    else:
        dataset = dataset[subgraph_index]
    return dataset


def load_layer_GT(args):
    if args.format == "csr":
        layer = SparseMHA_CSR(args.dim, args.dim, args.heads)
    elif args.format == "csr_gm":
        layer = SparseMHA_CSR_GM(args.dim, args.dim, args.heads)
    elif args.format == "tiling":
        layer = SparseMHA_tiling(args.dim, args.dim, args.heads)
    elif args.format == "hyper" or args.format == "nofuse":
        layer = SparseMHA_hyper(args.dim, args.dim, args.heads)
    elif args.format == "softmax":
        layer = SparseMHA_softmax(args.dim, args.dim, args.heads)
    elif args.format == "softmax_gm":
        layer = SparseMHA_softmax_gm(args.dim, args.dim, args.heads)
    # elif args.format == "indegree":
    #     layer = SparseMHA_indegree(args.dim, args.dim, args.heads)
    # elif args.format == "indegree_hyper":
    #     layer = SparseMHA_indegree_hyper(args.dim, args.dim, args.heads)
    # elif args.format == "subgraph":
    #     layer = SparseMHA_subgraph(args.dim, args.dim, args.heads)
    elif args.format == "hybrid":
        layer = SparseMHA_hybrid(args.dim, args.dim, args.heads)
    elif args.format == "forward":
        layer = SparseMHA_forward_timing(args.dim, args.dim, args.heads)
    elif args.format == "hyper_ablation":
        layer = SparseMHA_hyper_ablation(args.dim, args.dim, args.heads)
    elif args.format == "pyg":
        layer = SparseMHA_pyg(args.dim, args.dim, args.heads)
    elif args.format == "cugraph":
        layer = SparseMHA_cugraph(args.dim, args.dim, args.heads)
    else:
        raise ValueError(f"Unsupported format {args.format} in GTconv")
    return layer


def load_layer_GAT(args):
    if args.format == "csr":
        layer = GATConv_dgNN(args.dim, args.dim, args.heads)
    elif args.format == "tiling":
        layer = GATConv_tiling(args.dim, args.dim, args.heads)
    elif args.format == "hyper" or args.format == "nofuse":
        layer = GATConv_hyper(args.dim, args.dim, args.heads)
    elif args.format == "hyper_v2":
        layer = GATConv_hyper_v2(args.dim, args.dim, args.heads)
    elif args.format == "hyper_recompute":
        layer = GATConv_hyper_recompute(args.dim, args.dim, args.heads)
    elif args.format == "softmax":
        layer = GATConv_softmax(args.dim, args.dim, args.heads)
    elif args.format == "softmax_gm":
        layer = GATConv_softmax_gm(args.dim, args.dim, args.heads)
    elif args.format == "hybrid":
        layer = GATConv_hybrid(args.dim, args.dim, args.heads)
    elif args.format == "hyper_ablation":
        layer = GATConv_hyper_ablation(args.dim, args.dim, args.heads)
    elif args.format == "pyg":
        layer = GATConv_pyg(args.dim, args.dim, args.heads)
    elif args.format == "cugraph":
        layer = GATConv_cugraph(args.dim, args.dim, args.heads)
    else:
        raise ValueError(f"Unsupported format {args.format} in GATconv")
    return layer


def load_layer_AGNN(args):
    if args.format == "hyper" or args.format == "nofuse":
        layer = AGNNConv_hyper(args.dim, args.dim, args.heads)
    elif args.format == "csr":
        layer = AGNNConv_csr(args.dim, args.dim, args.heads)
    elif args.format == "softmax":
        layer = AGNNConv_softmax(args.dim, args.dim, args.heads)
    elif args.format == "csr_gm":
        layer = AGNNConv_csr_gm(args.dim, args.dim, args.heads)
    elif args.format == "tiling":
        layer = AGNNConv_tiling(args.dim, args.dim, args.heads)
    elif args.format == "softmax_gm":
        layer = AGNNConv_softmax_gm(args.dim, args.dim, args.heads)
    elif args.format == "pyg":
        layer = AGNNConv_pyg(args.dim, args.dim, args.heads)
    elif args.format == "cugraph":
        layer = AGNNConv_cugraph(args.dim, args.dim, args.heads)
    else:
        raise ValueError(f"Unsupported format {args.format} in AGNNconv")
    return layer


def load_graphconv_layer(args):
    if args.conv == "gat":
        layer = load_layer_GAT(args)
    elif args.conv == "gt":
        layer = load_layer_GT(args)
    elif args.conv == "agnn":
        layer = load_layer_AGNN(args)
    else:
        raise ValueError(f"unknown graph conv {args.conv}")
    return layer


def load_prepfunc(args):
    if args.format in ["csr", "csr_gm", "tiling"]:
        preprocess_func = preprocess_CSR
    elif args.format in [
        "hyper",
        "nofuse",
        "hyper_ablation",
        "hyper_recompute",
        "hyper_v2",
    ]:
        preprocess_func = preprocess_Hyper
    elif args.format in ["softmax", "softmax_gm"]:
        preprocess_func = preprocess_softmax
    # elif args.format == "indegree":
    #     preprocess_func = preprocess_indegree
    # elif args.format == "indegree_hyper":
    #     preprocess_func = preprocess_indegree_hyper
    elif args.format == "subgraph":
        preprocess_func = preprocess_SubGraph
    elif args.format == "hybrid":
        preprocess_func = preprocess_hybrid
    elif args.format == "forward":
        preprocess_func = preprocess_Hyper_fw_bw
    elif args.format == "pyg":
        preprocess_func = preprocess_pyg
    elif args.format == "cugraph":
        if args.conv in ["gt", "agnn"]:
            preprocess_cugraph_bip = partial(preprocess_cugraph, is_bipartite=True)
            preprocess_func = preprocess_cugraph_bip
        else:
            preprocess_func = preprocess_cugraph
    else:
        raise ValueError(f"Unsupported format {args.format}")
    return preprocess_func

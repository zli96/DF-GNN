import warnings

import dgl.sparse as dglsp
import torch

from dgl.data import Subset

from .gfconv_layer import SparseMHA
from .gfconv_layer_hyper import SparseMHA_hyper, SparseMHA_hyper_nofuse
from .gfconv_layer_subgraph import SparseMHA_outdegree, SparseMHA_subgraph

WARP_SIZE = 32


def g_to_SPmatrix(g):
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    M = g.num_edges()
    val = torch.ones(M) * 1.5
    A = dglsp.spmatrix(indices, val=val, shape=(N, N))
    return A


def preprocess_CSR(g, **args):
    A = g_to_SPmatrix(g)

    # using max_degree to cal max smem consume
    max_degree = int(max(A.sum(1)).item())
    smem_consume = (max_degree + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    print("preprocess smem consume", smem_consume)

    # the CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    return A, row_ptr, col_ind, val, smem_consume


def preprocess_Hyper(g, **args):
    A = g_to_SPmatrix(g)

    # using max_degree to cal max smem consume
    max_degree = int(max(A.sum(1)).item())
    smem_consume = (max_degree * 8 + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    print("preprocess smem consume", smem_consume)

    # A.row: the src node of each edge
    rows = A.row.int()
    rows = torch.sort(rows).values

    # the CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    return A, row_ptr, col_ind, rows, val, smem_consume


def preprocess_Hyper_nofuse(g, **args):
    A = g_to_SPmatrix(g)

    # using max_degree to cal max smem consume
    max_degree = int(max(A.sum(1)).item())
    smem_consume = (max_degree + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    print("preprocess smem consume", smem_consume)

    # A.row: the src node of each edge
    rows = A.row.int()
    rows = torch.sort(rows).values

    # the CSR format of adj matrix
    row_ptr, col_ind, val_idx = A.csr()
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = A.val[val_idx]
    return A, row_ptr, col_ind, rows, val, smem_consume


def preprocess_SubGraph(g, **args):
    nodes = g.batch_num_nodes()

    # num of nodes in each sub-graph(accumulate)
    nodes_subgraph = torch.cat(
        (torch.tensor([0]), torch.cumsum(nodes.clone(), 0))
    ).int()
    A = g_to_SPmatrix(g)

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


def preprocess_Outdegree(g, dim):
    ## global graph config
    # nodes_subgraph: num of nodes in each sub-graph(accumulate), shape(num_subgraph+1)
    nodes = g.batch_num_nodes()
    nodes_subgraph = torch.cat(
        (torch.tensor([0]), torch.cumsum(nodes.clone(), 0))
    ).int()
    A = g_to_SPmatrix(g)
    N = A.shape[0]
    out_degree = A.sum(0).int()
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
        out_degree_local = out_degree[node_lb:node_hb]
        degree_limit = sum(out_degree_local > 1).item()
        max_nodes_local = min(max_nodes, degree_limit)
        cumsum += max_nodes_local
        smem_nodes_subgraph.append(cumsum)
        _, indices = torch.sort(out_degree_local, descending=True)
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
    # val = A.val[val_idx]

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


def load_layer_prepfunc(args):
    if args.format == "csr":
        layer = SparseMHA
        preprocess_func = preprocess_CSR
    elif args.format == "hyper":
        layer = SparseMHA_hyper
        preprocess_func = preprocess_Hyper
    elif args.format == "hyper_nofuse":
        layer = SparseMHA_hyper_nofuse
        preprocess_func = preprocess_Hyper_nofuse
    elif args.format == "outdegree":
        layer = SparseMHA_outdegree
        preprocess_func = preprocess_Outdegree
    elif args.format == "subgraph":
        layer = SparseMHA_subgraph
        preprocess_func = preprocess_SubGraph
    else:
        raise ValueError(f"Unsupported format {args.format}")
    return layer, preprocess_func
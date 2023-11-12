# import dgNN

import fused_gtconv as fused_gt
import torch
from torch.utils.cpp_extension import load


def GTConvFuse_inference_indegree_hyper(
    row,
    row_ptr,
    col_ind,
    val,
    nodes_subgraph,
    smem_nodes_subgraph,
    store_node,
    store_flag,
    Q,
    K,
    V,
):
    out_feat = fused_gt.gt_indegree_hyper_forward(
        row,
        row_ptr,
        col_ind,
        val,
        nodes_subgraph,
        smem_nodes_subgraph,
        store_node,
        store_flag,
        Q,
        K,
        V,
    )
    return out_feat[0]


def GTConvFuse_inference_indegree(
    row_ptr,
    col_ind,
    val,
    nodes_subgraph,
    smem_nodes_subgraph,
    store_node,
    store_flag,
    Q,
    K,
    V,
):
    out_feat = fused_gt.gt_indegree_forward(
        row_ptr,
        col_ind,
        val,
        nodes_subgraph,
        smem_nodes_subgraph,
        store_node,
        store_flag,
        Q,
        K,
        V,
    )
    return out_feat[0]


def GTConvFuse_inference_subgraph(
    nodes_subgraph,
    indptr,
    indices,
    val,
    Q,
    K,
    V,
):
    out_feat = fused_gt.gt_subgraph_forward(
        nodes_subgraph,
        indptr,
        indices,
        val,
        Q,
        K,
        V,
    )
    return out_feat[0]


def GTConvFuse_inference_hyper(
    indptr,
    indices,
    rows,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    out_feat = fused_gt.gt_hyper_forward(
        indptr,
        indices,
        rows,
        val,
        smem_consume,
        Q,
        K,
        V,
    )
    return out_feat[0]


# class FusedGTFunction_hyper(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx,
#         indptr,
#         indices,
#         rows,
#         val,
#         smem_consume,
#         Q,
#         K,
#         V,
#     ):
#         out_feat = fused_gt.gt_hyper_forward(
#             indptr,
#             indices,
#             rows,
#             val,
#             smem_consume,
#             Q,
#             K,
#             V,
#         )
#         return out_feat[0]


def GTConvFuse_inference_softmax(
    indptr,
    indices,
    rows,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    out_feat = fused_gt.gt_softmax_forward(
        indptr,
        indices,
        rows,
        val,
        smem_consume,
        Q,
        K,
        V,
    )
    return out_feat[0]


def GTConvFuse_inference_csr(
    indptr,
    indices,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    out_feat = fused_gt.gt_csr_forward(
        indptr,
        indices,
        val,
        smem_consume,
        Q,
        K,
        V,
    )

    return out_feat[0]

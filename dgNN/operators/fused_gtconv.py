# import dgNN
import pdb

import fused_gtconv as fused_gt
import torch
from torch.utils.cpp_extension import load


def GTConvFuse_indegree_hyper(
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


def GTConvFuse_indegree(
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
    return FusedGTFunction_indegree.apply(
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


class FusedGTFunction_indegree(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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


def GTConvFuse_subgraph(
    nodes_subgraph,
    indptr,
    indices,
    val,
    Q,
    K,
    V,
):
    return FusedGTFunction_subgraph.apply(
        nodes_subgraph,
        indptr,
        indices,
        val,
        Q,
        K,
        V,
    )


class FusedGTFunction_subgraph(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
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


def GTConvFuse_hyper(
    indptr,
    indices,
    rows,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    return FusedGTFunction_hyper.apply(
        indptr,
        indices,
        rows,
        val,
        smem_consume,
        Q,
        K,
        V,
    )


class FusedGTFunction_hyper(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indptr,
        indices,
        rows,
        val,
        smem_consume,
        Q,
        K,
        V,
    ):
        out_feat = fused_gt.gt_hyper_fused_forward(
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


def GTConvFuse_hyper_nofuse(
    indptr,
    indices,
    rows,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    return FusedGTFunction_hyper_nofuse.apply(
        indptr,
        indices,
        rows,
        val,
        smem_consume,
        Q,
        K,
        V,
    )


class FusedGTFunction_hyper_nofuse(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indptr,
        indices,
        rows,
        val,
        smem_consume,
        Q,
        K,
        V,
    ):
        out_feat = fused_gt.gt_hyper_nofuse_forward(
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


def GTConvFuse(
    indptr,
    indices,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    return FusedGTFunction.apply(
        indptr,
        indices,
        val,
        smem_consume,
        Q,
        K,
        V,
    )


class FusedGTFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indptr,
        indices,
        val,
        smem_consume,
        Q,
        K,
        V,
    ):
        out_feat = fused_gt.gt_forward(
            indptr,
            indices,
            val,
            smem_consume,
            Q,
            K,
            V,
        )
        # ctx.save_for_backward(
        #     indptr,
        #     indices,
        #     col_ptr,
        #     row_ind,
        #     permute,
        #     edge_max,
        #     edge_sum,
        #     edge_mask,
        #     in_feat,
        #     attn_row,
        #     attn_col,
        # )
        return out_feat[0]

    # @staticmethod
    # def backward(ctx, grad_out):
    #     (
    #         indptr,
    #         indices,
    #         col_ptr,
    #         row_ind,
    #         permute,
    #         edge_max,
    #         edge_sum,
    #         edge_mask,
    #         in_feat,
    #         attn_row,
    #         attn_col,
    #     ) = ctx.saved_tensors
    #     grad_out = grad_out.contiguous()
    #     # print('start backward')
    #     grad_feat, grad_attn_row, grad_attn_col = fused_gat.gat_backward(
    #         ctx.negative_slope,
    #         ctx.attn_drop,
    #         indptr,
    #         indices,
    #         col_ptr,
    #         row_ind,
    #         permute,
    #         edge_max,
    #         edge_sum,
    #         edge_mask,
    #         in_feat,
    #         attn_row,
    #         attn_col,
    #         grad_out,
    #     )
    #     # print('end backward')
    #     # print(torch.isnan(grad_feat).sum())
    #     # print(torch.isnan(grad_attn_row).sum())
    #     # print(torch.isnan(grad_attn_col).sum())
    #     return (
    #         grad_attn_row,
    #         grad_attn_col,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         None,
    #         grad_feat,
    #         None,
    #     )

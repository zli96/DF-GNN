# import dgNN
import pdb

import fused_gfconv as fused_gf
import torch
from torch.utils.cpp_extension import load


def GFConvFuse_indegree(
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
    return FusedGFFunction_indegree.apply(
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


class FusedGFFunction_indegree(torch.autograd.Function):
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
        out_feat = fused_gf.gf_indegree_forward(
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


def GFConvFuse_subgraph(
    nodes_subgraph,
    indptr,
    indices,
    val,
    Q,
    K,
    V,
):
    return FusedGFFunction_subgraph.apply(
        nodes_subgraph,
        indptr,
        indices,
        val,
        Q,
        K,
        V,
    )


class FusedGFFunction_subgraph(torch.autograd.Function):
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
        out_feat = fused_gf.gf_subgraph_forward(
            nodes_subgraph,
            indptr,
            indices,
            val,
            Q,
            K,
            V,
        )
        return out_feat[0]


def GFConvFuse_hyper(
    indptr,
    indices,
    rows,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    return FusedGFFunction_hyper.apply(
        indptr,
        indices,
        rows,
        val,
        smem_consume,
        Q,
        K,
        V,
    )


class FusedGFFunction_hyper(torch.autograd.Function):
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
        out_feat = fused_gf.gf_hyper_fused_forward(
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


def GFConvFuse_hyper_nofuse(
    indptr,
    indices,
    rows,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    return FusedGFFunction_hyper_nofuse.apply(
        indptr,
        indices,
        rows,
        val,
        smem_consume,
        Q,
        K,
        V,
    )


class FusedGFFunction_hyper_nofuse(torch.autograd.Function):
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
        out_feat = fused_gf.gf_hyper_nofuse_forward(
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


def GFConvFuse(
    indptr,
    indices,
    val,
    smem_consume,
    Q,
    K,
    V,
):
    return FusedGFFunction.apply(
        indptr,
        indices,
        val,
        smem_consume,
        Q,
        K,
        V,
    )


class FusedGFFunction(torch.autograd.Function):
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
        out_feat = fused_gf.gf_forward(
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


def GFConvFuse_ELL(
    indptr,
    indices,
    row_index,
    rows_per_tb,
    val,
    Q,
    K,
    V,
):
    return FusedGFFunction_ELL.apply(
        indptr,
        indices,
        row_index,
        rows_per_tb,
        val,
        Q,
        K,
        V,
    )


class FusedGFFunction_ELL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indptr,
        indices,
        row_index,
        rows_per_tb,
        val,
        Q,
        K,
        V,
    ):
        out_feat = fused_gf.gf_ell_forward(
            indptr,
            indices,
            row_index,
            rows_per_tb,
            val,
            Q,
            K,
            V,
        )
        return out_feat[0]

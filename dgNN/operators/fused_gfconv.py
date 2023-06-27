# import dgNN
import fused_gfconv as fused_gf
import torch
from torch.utils.cpp_extension import load
import pdb


def GFConvFuse(
    row_ptr,
    col_ind,
    val,
    Q,
    K,
    V,
):
    return FusedGFFunction.apply(
        row_ptr,
        col_ind,
        val,
        Q,
        K,
        V,
    )


class FusedGFFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        row_ptr,
        col_ind,
        val,
        Q,
        K,
        V,
    ):
        out_feat = fused_gf.gf_forward(
            row_ptr,
            col_ind,
            val,
            Q,
            K,
            V,
        )
        # ctx.save_for_backward(
        #     row_ptr,
        #     col_ind,
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
    #         row_ptr,
    #         col_ind,
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
    #         row_ptr,
    #         col_ind,
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
    row_ptr,
    col_ind,
    row_index,
    rows_per_tb,
    val,
    Q,
    K,
    V,
):
    return FusedGFFunction_ELL.apply(
        row_ptr,
        col_ind,
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
        row_ptr,
        col_ind,
        row_index,
        rows_per_tb,
        val,
        Q,
        K,
        V,
    ):
        out_feat = fused_gf.gf_ell_forward(
            row_ptr,
            col_ind,
            row_index,
            rows_per_tb,
            val,
            Q,
            K,
            V,
        )
        return out_feat[0]
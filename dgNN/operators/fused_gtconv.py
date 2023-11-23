import fused_gtconv as fused_gt
import torch


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
    out_feat = fused_gt.gt_hyper_inference(
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


def GTConvFuse_hyper(
    rows,
    row_ptr,
    col_ind,
    val,
    col_ptr,
    row_ind,
    val_idx,
    smem_consume,
    Q,
    K,
    V,
):
    return FusedGTFunction_hyper.apply(
        rows,
        row_ptr,
        col_ind,
        val,
        col_ptr,
        row_ind,
        val_idx,
        smem_consume,
        Q,
        K,
        V,
    )


class FusedGTFunction_hyper(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rows,
        row_ptr,
        col_ind,
        val,
        col_ptr,
        row_ind,
        val_idx,
        smem_consume,
        Q,
        K,
        V,
    ):
        out_feat, attn_edge = fused_gt.gt_hyper_forward(
            row_ptr,
            col_ind,
            rows,
            val,
            col_ptr,
            row_ind,
            val_idx,
            smem_consume,
            Q,
            K,
            V,
        )
        ctx.smem = smem_consume
        ctx.save_for_backward(
            row_ptr, col_ind, rows, val, col_ptr, row_ind, val_idx, Q, K, V, attn_edge
        )
        return out_feat

    @staticmethod
    def backward(ctx, grad_out):
        (
            row_ptr,
            col_ind,
            rows,
            val,
            col_ptr,
            row_ind,
            val_idx,
            Q,
            K,
            V,
            attn_edge,
        ) = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        # print("start fused backward")
        # print(grad_out.shape)
        grad_Q, grad_K, grad_V = fused_gt.gt_backward(
            row_ptr,
            col_ind,
            rows,
            val,
            col_ptr,
            row_ind,
            val_idx,
            ctx.smem,
            Q,
            K,
            V,
            attn_edge,
            grad_out,
        )

        # print('end backward')
        # # print(torch.isnan(grad_feat).sum())
        # # print(torch.isnan(grad_attn_row).sum())
        # # print(torch.isnan(grad_attn_col).sum())
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_Q,
            grad_K,
            grad_V,
        )


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
    out_feat = fused_gt.gt_indegree_hyper_inference(
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
    out_feat = fused_gt.gt_indegree_inference(
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
    out_feat = fused_gt.gt_subgraph_inference(
        nodes_subgraph,
        indptr,
        indices,
        val,
        Q,
        K,
        V,
    )
    return out_feat[0]


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
    out_feat = fused_gt.gt_softmax_inference(
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
    out_feat = fused_gt.gt_csr_inference(
        indptr,
        indices,
        val,
        smem_consume,
        Q,
        K,
        V,
    )

    return out_feat[0]

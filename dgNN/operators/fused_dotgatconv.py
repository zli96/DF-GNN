# import dgNN
import fused_gtconv as fused_gt
import torch


def DOTGATConvFuse_hyper(indptr, indices, rows, val, smem_consume, H):
    return DOTGATFunction_hyper.apply(indptr, indices, rows, val, smem_consume, H)


class DOTGATFunction_hyper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indptr, indices, rows, val, smem_consume, H):
        out_feat = fused_gt.gt_hyper_fused_forward(
            indptr,
            indices,
            rows,
            val,
            smem_consume,
            H,
            H,
            H,
        )
        return out_feat[0]

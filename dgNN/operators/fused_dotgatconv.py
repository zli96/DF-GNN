import fused_dotgatconv as fused_dotgat
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


def DOTGATConvFuse_tile(indptr, indices, rows, smem_consume, H):
    return DOTGATFunction_tile.apply(indptr, indices, rows, smem_consume, H)


class DOTGATFunction_tile(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indptr, indices, val, smem_consume, H):
        out_feat = fused_dotgat.dotgat_tile_forward(
            indptr,
            indices,
            val,
            smem_consume,
            H,
        )
        return out_feat[0]
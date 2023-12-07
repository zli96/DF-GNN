import torch
from torch.nn import functional as F

from dgNN.operators.fused_gtconv import GTConvFuse_inference_hyper
from dgNN.utils import benchmark

from .agnn_layer import AGNNConvDGL


class AGNNConv_hyper(AGNNConvDGL):
    def conv(self, H, indptr, indices, rows, val, smem_consume):
        H_norm = F.normalize(H, p=2, dim=-1)
        out = GTConvFuse_inference_hyper(
            indptr,
            indices,
            rows,
            val,
            smem_consume,
            H_norm,
            H_norm,
            H,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        g, indptr, indices, rows, val, smem_consume = params
        H = self.proj(feat).view(-1, self.num_heads, self.out_size)
        val = torch.full_like(val, self.conv_nofuse.beta[0].item())
        if fuse:
            H = H.contiguous()
            out, elapsed_time = benchmark(
                self.conv,
                H,
                indptr,
                indices,
                rows,
                val,
                smem_consume,
            )
        else:
            out, elapsed_time = benchmark(self.forward_nofuse, g, feat)
        return out.reshape(N, -1), elapsed_time * 1000

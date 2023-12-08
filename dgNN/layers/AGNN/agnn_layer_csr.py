from torch.nn import functional as F

from dgNN.operators.fused_gtconv import GTConvFuse_inference_csr
from dgNN.utils import benchmark

from .agnn_layer import AGNNConvDGL


class AGNNConv_csr(AGNNConvDGL):
    def conv(self, H, indptr, indices, val, smem_consume):
        H_norm = F.normalize(H, p=2, dim=-1)
        out = GTConvFuse_inference_csr(
            indptr,
            indices,
            val,
            smem_consume,
            H_norm,
            H_norm,
            H,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        A, indptr, indices, val, smem_consume = params
        H = self.proj(feat).view(-1, self.num_heads, self.out_size)
        if fuse:
            H = H.contiguous()
            out, elapsed_time = benchmark(
                self.conv,
                H,
                indptr,
                indices,
                val,
                smem_consume,
            )
        else:
            H = H.reshape(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_nofuse, A, H)
            out = out.transpose(1, 2)
        return out.reshape(N, -1), elapsed_time * 1000

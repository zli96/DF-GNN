from DFGNN.operators.fused_gtconv import GTConvFuse_inference_tiling
from DFGNN.utils import benchmark

from .gtconv_layer import SparseMHA


class SparseMHA_tiling(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q, k, v = self.prep_qkv(h)

        if fuse:
            indptr, indices, val, smem_consume = params
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_tiling, indptr, indices, val, smem_consume, q, k, v
            )
            out = out.transpose(1, 2)
        else:
            A = params
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000

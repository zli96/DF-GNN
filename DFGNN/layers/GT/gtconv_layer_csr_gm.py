from DFGNN.operators.fused_gtconv import GTConvFuse_inference_csr_gm
from DFGNN.utils import benchmark

from .gtconv_layer import SparseMHA

# feature parallel, use global memory to store intermediate value
class SparseMHA_CSR_GM(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q, k, v = self.prep_qkv(h)

        if fuse:
            indptr, indices, val, _ = params
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_csr_gm, indptr, indices, val, q, k, v
            )
            out = out.transpose(1, 2)
        else:
            A = params
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000

from DFGNN.operators.fused_gtconv import GTConvFuse_inference_csr_gm
from DFGNN.utils import benchmark

from .gtconv_layer import SparseMHA

# feature parallel, use global memory to store intermediate value
class SparseMHA_CSR_GM(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        h = self.in_proj(h)

        ## get Q, K, V features
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, indptr, indices, val, _ = params

        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_csr_gm, indptr, indices, val, q, k, v
            )
            out = out.transpose(1, 2)
        else:
            out, elapsed_time = benchmark(self.forward_nofuse, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000

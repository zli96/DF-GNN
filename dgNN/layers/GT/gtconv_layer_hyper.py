import pdb

from dgNN.operators.fused_gtconv import GTConvFuse_hyper, GTConvFuse_hyper_nofuse
from dgNN.utils import benchmark

from .gtconv_layer import SparseMHA


class SparseMHA_hyper(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, indptr, indices, rows, val, smem_consume = params

        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_hyper, indptr, indices, rows, val, smem_consume, q, k, v
            )
            out = out.transpose(1, 2)
        else:
            out, elapsed_time = self.forward_nofuse(N, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000


class SparseMHA_hyper_nofuse(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, indptr, indices, rows, val, smem_consume = params

        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_hyper_nofuse,
                indptr,
                indices,
                rows,
                val,
                smem_consume,
                q,
                k,
                v,
            )
            out = out.transpose(1, 2)
        else:
            out, elapsed_time = self.forward_nofuse(N, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000
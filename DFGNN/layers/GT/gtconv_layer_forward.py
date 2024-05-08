from DFGNN.operators.fused_gtconv import GTConvFuse_hyper, GTConvFuse_inference_hyper
from DFGNN.utils import benchmark

from .gtconv_layer import SparseMHA


class SparseMHA_forward(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        (
            A,
            rows,
            row_ptr,
            col_ind,
            val,
            col_ptr,
            row_ind,
            val_idx,
            smem_consume,
        ) = params

        if fuse:
            q = self.q_proj(h).reshape(N, self.num_heads, self.head_dim)
            q *= self.scaling
            k = self.k_proj(h).reshape(N, self.num_heads, self.head_dim)
            v = self.v_proj(h).reshape(N, self.num_heads, self.head_dim)
            if self.training:
                out = GTConvFuse_hyper(
                    rows,
                    row_ptr,
                    col_ind,
                    val,
                    col_ptr,
                    row_ind,
                    val_idx,
                    smem_consume,
                    q,
                    k,
                    v,
                )
            else:
                out = GTConvFuse_inference_hyper(
                    row_ptr,
                    col_ind,
                    rows,
                    val,
                    smem_consume,
                    q,
                    k,
                    v,
                )
        else:
            ## get Q, K, V features
            q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
            q *= self.scaling
            k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
            v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
            out = self.forward_dglsp(A, q, k, v)
        return out.reshape(N, -1)


class SparseMHA_forward_timing(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        (
            A,
            rows,
            row_ptr,
            col_ind,
            val,
            col_ptr,
            row_ind,
            val_idx,
            smem_consume,
        ) = params

        if fuse:
            q = self.q_proj(h).reshape(N, self.num_heads, self.head_dim)
            q *= self.scaling
            k = self.k_proj(h).reshape(N, self.num_heads, self.head_dim)
            v = self.v_proj(h).reshape(N, self.num_heads, self.head_dim)
            out, elapsed_time = benchmark(
                GTConvFuse_hyper,
                rows,
                row_ptr,
                col_ind,
                val,
                col_ptr,
                row_ind,
                val_idx,
                smem_consume,
                q,
                k,
                v,
            )
            out = out.transpose(1, 2)
        else:
            ## get Q, K, V features
            q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
            q *= self.scaling
            k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
            v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)
        return out.reshape(N, -1), elapsed_time * 1000

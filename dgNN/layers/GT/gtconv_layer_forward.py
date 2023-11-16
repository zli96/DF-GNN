from dgNN.operators.fused_gtconv import GTConvFuse_hyper

from .gtconv_layer import SparseMHA


class SparseMHA_fused(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
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
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
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

            out = out.transpose(1, 2)

        else:
            out = self.forward_nofuse(A, q, k, v)
        return self.out_proj(out.reshape(N, -1))

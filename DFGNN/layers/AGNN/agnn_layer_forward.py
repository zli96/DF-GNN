from torch.nn import functional as F

from DFGNN.operators.fused_gtconv import GTConvFuse_hyper

from .agnn_layer import AGNNConvDGL


class AGNNConv_forward(AGNNConvDGL):
    def conv(
        self,
        H,
        rows,
        row_ptr,
        col_ind,
        val,
        col_ptr,
        row_ind,
        val_idx,
        smem_consume,
    ):
        H_norm = F.normalize(H, p=2, dim=-1)
        out = GTConvFuse_hyper(
            rows,
            row_ptr,
            col_ind,
            val,
            col_ptr,
            row_ind,
            val_idx,
            smem_consume,
            H_norm,
            H_norm,
            H,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
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
            H = self.proj(feat).view(-1, self.num_heads, self.out_size)
            out = self.conv(
                H,
                rows,
                row_ptr,
                col_ind,
                val,
                col_ptr,
                row_ind,
                val_idx,
                smem_consume,
            )
        else:
            H = self.proj(feat).view(-1, self.out_size, self.num_heads)
            out = self.forward_nofuse(A, H)
        return out.reshape(N, -1)

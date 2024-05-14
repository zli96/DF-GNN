from DFGNN.operators.fused_gatconv import GATConvFuse_inference_tiling
from DFGNN.utils import benchmark

from .gatconv_layer import GATConvDGL

# our gat layer in hyper method
class GATConv_tiling(GATConvDGL):
    def conv(self, row_ptr, col_ind, a_l, a_r, h):
        attn_row = (a_l * h).sum(dim=-1)
        attn_col = (a_r * h).sum(dim=-1)
        out = GATConvFuse_inference_tiling(
            attn_row, attn_col, row_ptr, col_ind, self.negative_slope, h
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        if fuse:
            row_ptr, col_ind, _, _ = params
            feat = self.W(feat).view(-1, self.num_heads, self.out_size)
            out, elapsed_time = benchmark(
                self.conv,
                row_ptr,
                col_ind,
                self.a_l.transpose(1, 2),
                self.a_r.transpose(1, 2),
                feat,
            )
        else:
            A = params
            feat = self.W(feat).view(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000

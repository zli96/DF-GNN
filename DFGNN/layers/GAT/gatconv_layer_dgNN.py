from DFGNN.operators.fused_gatconv import GATConvFuse_inference
from DFGNN.utils import benchmark

from .gatconv_layer import GATConvDGL

# our gat layer in hyper method
class GATConv_dgNN(GATConvDGL):
    def conv(self, row_ptr, col_ind, feat):
        h = self.W(feat).view(-1, self.out_size, self.num_heads)
        attn_row = (self.a_l * h).sum(dim=1)
        attn_col = (self.a_r * h).sum(dim=1)
        h = h.view(-1, self.num_heads, self.out_size)

        out = GATConvFuse_inference(
            attn_row, attn_col, row_ptr, col_ind, self.negative_slope, h
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        A, row_ptr, col_ind, _, _ = params
        if fuse:
            out, elapsed_time = benchmark(self.conv, row_ptr, col_ind, feat)
        else:
            out, elapsed_time = benchmark(self.forward_nofuse, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000

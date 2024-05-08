from DFGNN.operators.fused_gatconv import GATConvFuse_inference_softmax
from DFGNN.utils import benchmark

from .gatconv_layer import GATConvDGL


class GATConv_softmax(GATConvDGL):
    def conv(self, indptr, indices, rows, smem_consume, feat):
        h = self.W(feat).view(-1, self.out_size, self.num_heads)
        attn_row = (self.a_l * h).sum(dim=1)
        attn_col = (self.a_r * h).sum(dim=1)
        h = h.view(-1, self.num_heads, self.out_size)
        out = GATConvFuse_inference_softmax(
            smem_consume,
            attn_row,
            attn_col,
            indptr,
            indices,
            rows,
            self.negative_slope,
            h,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        A, indptr, indices, rows, _, smem_consume = params
        if fuse:
            out, elapsed_time = benchmark(
                self.conv, indptr, indices, rows, smem_consume, feat
            )
        else:
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000

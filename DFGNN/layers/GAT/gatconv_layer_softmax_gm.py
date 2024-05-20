from DFGNN.operators.fused_gatconv import GATConvFuse_inference_softmax_gm
from DFGNN.utils import benchmark

from .gatconv_layer import GATConvDGL


class GATConv_softmax_gm(GATConvDGL):
    def conv(self, indptr, indices, rows, a_l, a_r, h):
        attn_row = (a_l * h).sum(dim=-1)
        attn_col = (a_r * h).sum(dim=-1)
        out = GATConvFuse_inference_softmax_gm(
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
        if fuse:
            indptr, indices, rows, _, _ = params
            feat = self.W(feat).view(-1, self.num_heads, self.out_size)
            out, elapsed_time = benchmark(
                self.conv,
                indptr,
                indices,
                rows,
                self.a_l.transpose(1, 2),
                self.a_r.transpose(1, 2),
                feat,
            )
        else:
            A = params
            feat = self.W(feat).view(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000

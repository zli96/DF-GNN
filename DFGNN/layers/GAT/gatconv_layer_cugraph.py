import torch
from pylibcugraphops.pytorch.operators import mha_gat_n2n

from DFGNN.utils import benchmark

from .gatconv_layer import GATConvDGL

# our gat layer in hyper method
class GATConv_cugraph(GATConvDGL):
    def forward(self, params, feat, fuse=False):
        N = len(feat)
        if fuse:
            graph = params
            feat = self.W(feat)
            attn_weights = torch.concat([self.a_r, self.a_l]).flatten()
            out, elapsed_time = benchmark(
                mha_gat_n2n,
                feat,
                attn_weights,
                graph,
                self.num_heads,
                "LeakyReLU",
                self.negative_slope,
                True,
            )
        else:
            A = params
            feat = self.W(feat).view(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000

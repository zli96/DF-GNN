from dgNN.operators.fused_dotgatconv import DOTGATConvFuse_hyper
from dgNN.utils import benchmark

from .dotgatconv_layer import DOTGATConvDGL


class DOTGATConv_hyper(DOTGATConvDGL):
    def forward(self, params, feat, fuse=False):
        N = len(feat)
        g, indptr, indices, rows, val, smem_consume = params
        H = self.conv_nofuse.fc(feat).view(-1, self.num_heads, self.out_size)

        if fuse:
            feat = feat.contiguous()
            out, elapsed_time = benchmark(
                DOTGATConvFuse_hyper, indptr, indices, rows, val, smem_consume, H
            )
        else:
            out, elapsed_time = self.forward_nofuse(N, g, feat)
        return out.reshape(N, -1), elapsed_time * 1000

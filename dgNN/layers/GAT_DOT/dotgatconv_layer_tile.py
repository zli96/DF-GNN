from dgNN.operators.fused_dotgatconv import DOTGATConvFuse_tile
from dgNN.utils import benchmark

from .dotgatconv_layer import DOTGATConvDGL


class DOTGATConv_tile(DOTGATConvDGL):
    def forward(self, params, feat, fuse=False):
        N = len(feat)
        g, indptr, indices, val, smem_consume = params
        H = self.conv_nofuse.fc(feat).view(-1, self.num_heads, self.out_size)

        if fuse:
            feat = feat.contiguous()
            out, elapsed_time = benchmark(
                DOTGATConvFuse_tile, indptr, indices, val, smem_consume, H
            )
        else:
            out, elapsed_time = self.forward_nofuse(N, g, feat)
        return out.reshape(N, -1), elapsed_time * 1000
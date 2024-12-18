from DFGNN.operators.fused_gtconv import GTConvFuse_inference_hyper
from DFGNN.utils import benchmark

from .dotgatconv_layer import DOTGATConvDGL


class DOTGATConv_hyper(DOTGATConvDGL):
    def forward(self, params, feat, fuse=False):
        N = len(feat)
        g, indptr, indices, rows, val, smem_consume = params
        H = self.conv_nofuse.fc(feat).view(-1, self.num_heads, self.out_size)

        if fuse:
            feat = feat.contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_hyper,
                indptr,
                indices,
                rows,
                val,
                smem_consume,
                H,
                H,
                H,
            )
        else:
            out, elapsed_time = benchmark(self.forward_dglsp, g, feat)
        return out.reshape(N, -1), elapsed_time * 1000

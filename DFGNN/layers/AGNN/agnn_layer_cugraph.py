from pylibcugraphops.pytorch.operators import mha_simple_n2n

from torch.nn import functional as F

from DFGNN.utils import benchmark

from .agnn_layer import AGNNConvDGL


class AGNNConv_cugraph(AGNNConvDGL):
    def conv(self, H, N, graph):
        H_norm = F.normalize(H, p=2, dim=-1)
        H = H.reshape(N, -1)
        H_norm = H_norm.reshape(N, -1)
        out = mha_simple_n2n(
            H_norm,
            H_norm,
            H,
            graph,
            self.num_heads,
            True,
            None,
            False,
            None,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        H = self.proj(feat).view(-1, self.num_heads, self.out_size)
        if fuse:
            H = H.contiguous()
            graph = params
            out, elapsed_time = benchmark(self.conv, H, N, graph)
        else:
            A = params
            H = H.reshape(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, H)
            out = out.transpose(1, 2)
        return out.reshape(N, -1), elapsed_time * 1000

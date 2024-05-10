from pylibcugraphops.pytorch.operators import mha_simple_n2n

from DFGNN.utils import benchmark

from .gtconv_layer import SparseMHA


class SparseMHA_cugraph(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q, k, v = self.prep_qkv(h)
        if fuse:
            q = q.reshape(N, -1).contiguous()
            k = k.reshape(N, -1).contiguous()
            v = v.reshape(N, -1).contiguous()
            graph = params
            out, elapsed_time = benchmark(
                mha_simple_n2n,
                k,
                q,
                v,
                graph,
                self.num_heads,
                True,
                None,
                False,
                None,
            )
        else:
            A = params
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000

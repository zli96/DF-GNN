import dgl.function as fn
import dgl.sparse as dglsp
from dgl.nn.functional import edge_softmax

from DFGNN.utils import benchmark
from .gtconv_layer import SparseMHA


class SparseMHA_hybrid(SparseMHA):
    def forward_hybrid(self, A, A_csr, q, k, v):
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))
        attn = attn.softmax()
        out = dglsp.bspmm(A_csr, v)
        return out

    def forward_mpnn(self, graph, q, k, v):
        graph.dstdata.update({"q": q})
        graph.srcdata.update({"k": k})
        graph.srcdata.update({"v": v})

        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v("k", "q", "a"))

        # Step 2. edge softmax to compute attention scores
        graph.edata["sa"] = edge_softmax(graph, graph.edata["a"])

        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e("v", "sa", "attn"), fn.sum("attn", "agg_u"))

        # output results to the destination nodes
        rst = graph.dstdata["agg_u"]

        return rst

    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, A2 = params
        A2.csr()
        if fuse:
            out, elapsed_time = benchmark(
                self.forward_hybrid,
                A,
                A2,
                q,
                k,
                v,
            )
            # q = q.transpose(1, 2).contiguous()
            # k = k.transpose(1, 2).contiguous()
            # v = v.transpose(1, 2).contiguous()
            # out, elapsed_time = benchmark(
            #     self.forward_mpnn,
            #     g,
            #     q,
            #     k,
            #     v,
            # )
            # out = out.transpose(1, 2)
        else:
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000

from torch_geometric.nn import TransformerConv

from DFGNN.utils import benchmark

from .gtconv_layer import SparseMHA


class SparseMHA_pyg(SparseMHA):
    def __init__(self, in_size, out_size, num_heads):
        super().__init__(in_size, out_size, num_heads)
        self.conv = TransformerConv(
            in_channels=in_size,
            out_channels=out_size,
            heads=num_heads,
            root_weight=False,
        )

    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        if fuse:
            edge_index = params
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                self.conv.propagate, edge_index, q, k, v, None
            )
            out = out.transpose(1, 2)
        else:
            A = params
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000

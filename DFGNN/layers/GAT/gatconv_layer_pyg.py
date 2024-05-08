from torch import nn
from torch_geometric.nn import GATConv

from DFGNN.utils import benchmark

from .gatconv_layer import GATConvDGL


# our gat layer in hyper method
class GATConv_pyg(GATConvDGL):
    def __init__(self, in_size, out_size, num_heads, dropout=0, negative_slope=0.2):
        super().__init__(in_size, out_size, num_heads, dropout, negative_slope)
        self.conv = GATConv(
            in_channels=in_size,
            out_channels=out_size,
            dropout=dropout,
            negative_slope=negative_slope,
            bias=False,
            add_self_loops=False,
        )
        self.conv.lin = self.W
        self.conv.att_src = nn.Parameter(self.a_l.transpose(1, 2))
        self.conv.att_dst = nn.Parameter(self.a_r.transpose(1, 2))

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        if fuse:
            edge_index = params
            out, elapsed_time = benchmark(self.conv, feat, edge_index)
        else:
            A = params
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000

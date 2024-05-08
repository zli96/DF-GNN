from torch_geometric.nn import AGNNConv

from DFGNN.utils import benchmark

from .agnn_layer import AGNNConvDGL


class AGNNConv_pyg(AGNNConvDGL):
    def __init__(self, in_size, out_size, num_heads):
        super().__init__(in_size, out_size, num_heads)
        self.conv = AGNNConv(requires_grad=False)

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        H = self.proj(feat).view(-1, self.out_size)
        if fuse:
            edge_index = params
            H = H.contiguous()
            out, elapsed_time = benchmark(self.conv, H, edge_index)
        else:
            A = params
            H = H.reshape(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, H)
            out = out.transpose(1, 2)
        return out.reshape(N, -1), elapsed_time * 1000

from dgl.nn import DotGatConv
from dgNN.utils import Timer
from torch import nn


class DOTGATConvDGL(nn.Module):
    def __init__(self, in_size, out_size, num_heads):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.conv_nofuse = DotGatConv(in_size, out_size, num_heads)

    def forward_nofuse(self, N, g, feat):
        for i in range(3):
            out = self.conv_nofuse(g, feat)
        with Timer() as t:
            for i in range(100):
                out = self.conv_nofuse(g, feat)
        elapsed_time = t.elapsed_secs / 100
        return out.reshape(N, -1), elapsed_time

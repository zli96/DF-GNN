from dgl.nn import DotGatConv
from torch import nn


class DOTGATConvDGL(nn.Module):
    def __init__(self, in_size, out_size, num_heads):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.conv_nofuse = DotGatConv(in_size, out_size, num_heads)

    def forward_nofuse(self, g, feat):
        out = self.conv_nofuse(g, feat)
        return out

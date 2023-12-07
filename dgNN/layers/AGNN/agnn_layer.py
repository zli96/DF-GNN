from dgl.nn import AGNNConv
from torch import nn


class AGNNConvDGL(nn.Module):
    def __init__(self, in_size, out_size, num_heads):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.proj = nn.Linear(in_size, out_size)
        self.conv_nofuse = AGNNConv(learn_beta=False)

    def forward_nofuse(self, g, feat):
        feat = self.proj(feat)
        out = self.conv_nofuse(g, feat)
        return out

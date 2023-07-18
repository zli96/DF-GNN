import torch.nn as nn
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder

# For full graph dataset: cora
class GTlayer_fullgraph(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, in_size, hidden_size, num_heads=1):
        super().__init__()
        self.MHA = layer(hidden_size=hidden_size, num_heads=num_heads)
        self.linear = nn.Linear(in_size, hidden_size)

    def forward(self, params, X, fuse=False):
        h = self.linear(X)
        h = self.MHA(params, h, fuse)
        return h


# For dataset: ogbg-molhiv, ogbg-molpcba
class GTlayer_mol(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, hidden_size=2, num_heads=1):
        super().__init__()
        self.MHA = layer(hidden_size=hidden_size, num_heads=num_heads)
        self.atom_encoder = AtomEncoder(hidden_size)

    def forward(self, params, X, fuse=False):
        h = self.atom_encoder(X)
        h = self.MHA(params, h, fuse)
        return h

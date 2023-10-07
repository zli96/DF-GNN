import torch.nn as nn
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder

# For dataset with node feature: MNIST, CIFAR10, cora
class GTlayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, in_size, hidden_size, num_heads=1):
        super().__init__()
        self.MHA = layer(hidden_size=hidden_size, num_heads=num_heads)
        self.linear = nn.Linear(in_size, hidden_size)

    def forward(self, params, X, fuse=False):
        h = self.linear(X)
        h = self.MHA(params, h, fuse)
        return h


# For molcular dataset: ogbg-molhiv, ogbg-molpcba
class GTlayer_mol(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, hidden_size=64, num_heads=1):
        super().__init__()
        self.MHA = layer(hidden_size=hidden_size, num_heads=num_heads)
        self.atom_encoder = AtomEncoder(hidden_size)

    def forward(self, params, X, fuse=False):
        h = self.atom_encoder(X)
        h = self.MHA(params, h, fuse)
        return h


# For dataset need embedding: PATTERN, CLUSTER
class GTlayer_SBM(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, in_size=1, hidden_size=64, num_heads=1):
        super().__init__()
        self.MHA = layer(hidden_size=hidden_size, num_heads=num_heads)
        self.embedding_h = nn.Embedding(in_size, hidden_size)

    def forward(self, params, X, fuse=False):
        h = self.embedding_h(X)
        h = self.MHA(params, h, fuse)
        return h


def choose_GTlayer(dataset_name, MHAlayer, hidden_size, num_heads):
    if dataset_name in [
        "ogbg-molhiv",
        "ogbg-molpcba",
        "Peptides-func",
        "Peptides-struct",
    ]:
        return GTlayer_mol(MHAlayer, hidden_size, num_heads)
    elif dataset_name == "PATTERN":
        return GTlayer_SBM(MHAlayer, 3, hidden_size, num_heads)
    elif dataset_name == "CLUSTER":
        return GTlayer_SBM(MHAlayer, 7, hidden_size, num_heads)
    elif dataset_name == "MNIST":
        return GTlayer(MHAlayer, 3, hidden_size, num_heads)
    elif dataset_name == "CIFAR10":
        return GTlayer(MHAlayer, 5, hidden_size, num_heads)
    elif dataset_name in ["PascalVOC-SP", "COCO-SP"]:
        return GTlayer(MHAlayer, 14, hidden_size, num_heads)
    else:
        raise ValueError(f"unknown dataset {dataset_name}")

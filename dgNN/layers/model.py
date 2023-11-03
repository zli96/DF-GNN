import torch.nn as nn
import torch.nn.functional as F

from ogb.graphproppred.mol_encoder import AtomEncoder

# For dataset with node feature: MNIST, CIFAR10, cora
class Model(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, in_size, hidden_size, num_heads=1):
        super().__init__()
        self.MHA = layer
        self.linear = nn.Linear(in_size, hidden_size)

    def forward(self, params, X, fuse=False):
        h = self.linear(X)
        h = self.MHA(params, h, fuse)
        return h


# For molcular dataset: ogbg-molhiv, ogbg-molpcba
class Model_mol(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, hidden_size=64, num_heads=1):
        super().__init__()
        self.MHA = layer
        self.atom_encoder = AtomEncoder(hidden_size)

    def forward(self, params, X, fuse=False):
        h = self.atom_encoder(X)
        h = self.MHA(params, h, fuse)
        return h


# For dataset need embedding: PATTERN, CLUSTER
class Model_Emb(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, layer, in_size=1, hidden_size=64, num_heads=1):
        super().__init__()
        self.MHA = layer
        self.embedding_h = nn.Embedding(in_size, hidden_size)

    def forward(self, params, X, fuse=False):
        h = self.embedding_h(X)
        h = self.MHA(params, h, fuse)
        return h


def choose_Model(dataset_name, MHAlayer, hidden_size, num_heads):
    if dataset_name in [
        "ogbg-molhiv",
        "ogbg-molpcba",
        "Peptides-func",
        "Peptides-struct",
    ]:
        return Model_mol(MHAlayer, hidden_size, num_heads)
    elif dataset_name == "PATTERN":
        return Model_Emb(MHAlayer, 3, hidden_size, num_heads)
    elif dataset_name == "CLUSTER":
        return Model_Emb(MHAlayer, 7, hidden_size, num_heads)
    elif dataset_name == "MNIST":
        return Model(MHAlayer, 3, hidden_size, num_heads)
    elif dataset_name == "CIFAR10":
        return Model(MHAlayer, 5, hidden_size, num_heads)
    elif dataset_name in ["PascalVOC-SP", "COCO-SP"]:
        return Model(MHAlayer, 14, hidden_size, num_heads)
    else:
        raise ValueError(f"unknown dataset {dataset_name}")

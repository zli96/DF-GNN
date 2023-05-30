import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import AsGraphPredDataset
from ogb.graphproppred import collate_dgl, DglGraphPropPredDataset, Evaluator
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder

from dgNN.layers import SparseMHA

class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)
        self.atom_encoder = AtomEncoder(hidden_size)

    def forward(self, g, X, fuse=False):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        h = self.atom_encoder(X)
        h = self.MHA(A, h, fuse)

        return h


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    pos_enc_size = 8
    dataset = AsGraphPredDataset(
        DglGraphPropPredDataset("ogbg-molhiv", "./data/OGB")
    )
    evaluator = Evaluator("ogbg-molhiv")
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=256,
        collate_fn=collate_dgl,
    )
    
    out_size = dataset.num_tasks
    layer = GTLayer().to(dev)
    for batched_g, labels in train_dataloader:
        batched_g, labels = batched_g.to(dev), labels.to(dev)
        print("----------------------without fuse--------------------------")
        logits = layer(batched_g, batched_g.ndata["feat"])
        
        print("logits shape ", logits.shape)
        print("----------------------with fuse--------------------------")
        logits = layer(batched_g, batched_g.ndata["feat"],fuse=True)
        print("logits shape ", logits.shape)
        
        
        exit()
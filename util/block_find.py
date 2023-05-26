import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from torch.optim import Adam
import pdb

def block_density(A, i,j,block_size):
    block = A[i:i+block_size][j:j+block_size]
    return torch.count_nonzero(block).item()


def find_block(A, dev, block_size=2):
    conv = nn.Conv2d(1,1,(block_size, block_size),stride=2)
    
    with torch.no_grad():
        conv.weight = nn.Parameter(torch.ones((1,1,2,2)))
        conv.bias = nn.Parameter(torch.zeros((1)))
    conv.requires_grad_(False)
    conv = conv.to(dev)
    # conv.weight = torch.ones((1,1,2,2))
    # conv.bias = torch.zeros((1))
    A_group = conv(A.unsqueeze(0)).unsqueeze(0)
    upsample = nn.UpsamplingNearest2d(scale_factor=2)
    A_mask = upsample((A_group>2).float()).squeeze()
    print(torch.sum((A_group>2).float()), torch.sum(A_mask))
    
    print(A_mask.shape)
    print(A.shape)
    
    return  A_mask

def main(dataset):
    g = dataset[0].to(dev)
    # Create the adjacency matrix of graph.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    # find_block_sp(A)
    # pdb.set_trace()
    A_dense = A.to_dense().to(dev)
    A_mask = find_block(A_dense,dev)
    A_bsr = A_mask * A_dense
    print("block num: ", torch.sum(A_bsr).item())
    print("nnz=", A.nnz)                        


if __name__ == "__main__":
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load graph from the existing dataset.
    dataset = CoraGraphDataset()
    
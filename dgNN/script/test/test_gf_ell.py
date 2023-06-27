import argparse
import pdb

import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import format_conversion
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from ogb.graphproppred.mol_encoder import AtomEncoder

from dgNN.layers import SparseMHA_ELL
from dgNN.utils import load_data_full_graph


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=2, num_heads=1):
        super().__init__()
        self.MHA = SparseMHA_ELL(hidden_size=hidden_size, num_heads=num_heads)
        self.atom_encoder = AtomEncoder(hidden_size)
        
    def forward(self, params, X, ell=False):
        h = self.atom_encoder(X)
        h = self.MHA(params, h, ell)
        return h


def preprocess(
    g,
    bucket_sizes=[],
    num_col_parts=1,
):
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))
    row_ptr, col_ind, val_idx = A.csr()
    
    row_ptr = row_ptr.int()
    col_ind = col_ind.int()
    val = torch.tensor([A.val[i] for i in val_idx]).float()
    
    # cluster the rows into diff buckets based on its num of neighbors
    row_col_ind, _, _ = format_conversion.csr2ell(
        N, N, row_ptr, col_ind, num_col_parts, bucket_sizes
    )
    
    # num of elements each tb need to process
    elements_per_tb = 4
    rows_per_tb = []
    row_col_ind = row_col_ind[0]
    
    # calculate the num of elements each tb need to process
    for i, bucket_size in enumerate(bucket_sizes):
        num_elements = len(row_col_ind[i])
        num_rows = elements_per_tb//bucket_size
        rows_per_tb = rows_per_tb + [num_rows] *  (num_elements//num_rows)
        res = num_elements % num_rows
        if res !=0:
            rows_per_tb = rows_per_tb + [res]
    row_index = torch.cat(row_col_ind,0).int()
    rows_per_tb = torch.cat((torch.tensor([0]), torch.cumsum(torch.tensor(rows_per_tb),0))).int()
    
    # print(f"num_col_parts {num_col_parts}, bucket_sizes {bucket_sizes}")
    # print(f"graph nodes {N}, edges {nnz}")
    # print("--------------csr format----------------")
    # print(row_ptr)
    # print(col_ind)
    # print(row_index)
    # print(rows_per_tb)
    
    # pdb.set_trace()
    print("finish preprocess")
    return A, row_ptr, col_ind, row_index, rows_per_tb, val

col_part_config = {
    "arxiv": 1,
    "proteins": 8,
    "pubmed": 1,
    "citeseer": 1,
    "cora": 1,
    "ppi": 16,
    "reddit": 8,
    "products": 16,
}

bucketing_config = {
    "arxiv": [1, 2, 4, 8, 16, 32],
    "proteins": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "pubmed": [1, 2, 4, 8, 16, 32],
    "citeseer": [1, 2, 4],
    "cora": [1, 2, 4],
    "ppi": [1, 2, 4, 8, 16, 32],
    "products": [1, 2, 4, 8, 16, 32],
    "reddit": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("hybrid format spmm in sparse-tir")
    parser.add_argument(
        "--dataset", "-d", type=str, default="cora", help="dataset name"
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    print("a")
    
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    name = args.dataset
    layer = GTLayer(hidden_size=args.dim, num_heads=args.heads).to(dev)
    dataset = AsGraphPredDataset(DglGraphPropPredDataset("ogbg-molhiv", "./data/OGB"))
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=args.batch_size,
        collate_fn=collate_dgl,
        shuffle=False,
    )
    
    time_no_fuse = []
    time_fuse = []
    warmup = 5
    
    for i, (batched_g, labels) in enumerate(train_dataloader):
        params = preprocess(
            batched_g,
            bucket_sizes=bucketing_config[name],
            num_col_parts=col_part_config[name],
        )
        params = [param.to(dev) for param in params]
        batched_g, labels = batched_g.to(dev), labels.to(dev)
        logits, elapsed_time = layer(params, batched_g.ndata["feat"])
        if i > warmup:
            time_no_fuse.append(elapsed_time)
            print(f"epoch {i} non-fused time %.4f" % elapsed_time)
            # print("----------------------with fuse--------------------------")
            logits_fuse, elapsed_time = layer(params, batched_g.ndata["feat"], ell=True)
            time_fuse.append(elapsed_time)
            # pdb.set_trace()
            print(f"epoch {i} fused time %.4f" % elapsed_time)
            if all(torch.isclose(logits, logits_fuse, atol=0.001).flatten()):
                print("the results are the same, success!!!!!!!!!!")
            if i == 30:
                    break
    print("----------------------Result------------------------")
    print("no-fuse average time {:.4f} ms".format(sum(time_no_fuse) / len(time_no_fuse)))
    print("fuse average time {:.4f} ms".format(sum(time_fuse) / len(time_fuse)))


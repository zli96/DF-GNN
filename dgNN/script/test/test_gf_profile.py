import argparse
import pdb

import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AsGraphPredDataset
from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl
from ogb.graphproppred.mol_encoder import AtomEncoder

import ScheduleProfiler
profiler = ScheduleProfiler.ScheduleProfiler()

class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # [N, N, nh]
        attn = attn.softmax()
        out = dglsp.bspmm(attn, v)
        # torch.cuda.synchronize()
        # elapsed_time = time.time() - start
        elapsed_time = 0
        return out.reshape(N, -1), elapsed_time * 1000


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=2, num_heads=1):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.atom_encoder = AtomEncoder(hidden_size)

    def forward(self, g, X, fuse=False):
        indices = torch.stack(g.edges())
        N = g.num_nodes()
        A = dglsp.spmatrix(indices, shape=(N, N))
        h = self.atom_encoder(X)
        h = self.MHA(A, h, fuse)

        return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GF")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print("hidden dim", args.dim)
    print("num heads", args.heads)
    print("batch size", args.batch_size)
    # If CUDA is available, use GPU to accelerate the training, use CPU
    # otherwise.
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    dataset = AsGraphPredDataset(DglGraphPropPredDataset("ogbg-molhiv", "./data/OGB"))
    evaluator = Evaluator("ogbg-molhiv")
    train_dataloader = GraphDataLoader(
        dataset[dataset.train_idx],
        batch_size=args.batch_size,
        collate_fn=collate_dgl,
        shuffle=False,
    )

    out_size = dataset.num_tasks
    layer = GTLayer(hidden_size=args.dim, num_heads=args.heads).to(dev)
    
    print("----------------------Forward------------------------")
    time_no_fuse = []
    time_fuse = []
    warmup = 5
    # iter = 10 
    for i, (batched_g, labels) in enumerate(train_dataloader):
        batched_g, labels = batched_g.to(dev), labels.to(dev)
        profiler.start()
        # print("----------------------without fuse--------------------------")
        logits, elapsed_time = layer(batched_g, batched_g.ndata["feat"])
        profiler.stop()
        if i > warmup:
            time_no_fuse.append(elapsed_time)
            print(f"epoch {i} non-fused time %.4f" % elapsed_time)
            # print("----------------------with fuse--------------------------")
            # logits_fuse, elapsed_time = layer(batched_g, batched_g.ndata["feat"], fuse=True)
            # time_fuse.append(elapsed_time)
            # # pdb.set_trace()
            # print(f"epoch {i} fused time %.4f" % elapsed_time)
            # if all(torch.isclose(logits, logits_fuse, atol=0.001).flatten()):
            #     print("the results are the same, success!!!!!!!!!!")
            # else:
            #     for i in range(logits.shape[0]):
            #         if not all(torch.isclose(logits[i], logits_fuse[i], atol=0.001).flatten()):
            #             print(f"error node {i} mismatch")
            #             # print("neighbor nodes", col_ind[row_ptr[i]:row_ptr[i+1]])
            #             print(logits[i])
            #             print(logits_fuse[i])
            #             pdb.set_trace()

            if i == 30:
                break
    # print("----------------------Result------------------------")
    # print("no-fuse average time {:.4f} ms".format(sum(time_no_fuse) / len(time_no_fuse)))
    # print("fuse average time {:.4f} ms".format(sum(time_fuse) / len(time_fuse)))

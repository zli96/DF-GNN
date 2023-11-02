import pdb
import time

import dgl.sparse as dglsp
import torch
import torch.nn as nn

from dgNN.utils import benchmark, Timer


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

    def forward_nofuse(self, N, A, q, k, v):
        # TODO: spmm 要不要用csr格式（但好像更慢）
        # A.csr()
        for i in range(3):
            attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # [N, N, nh]
            attn = attn.softmax()
            out = dglsp.bspmm(attn, v)

        with Timer() as t:
            for i in range(100):
                attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # [N, N, nh]
                attn = attn.softmax()
                out = dglsp.bspmm(attn, v)
        elapsed_time = t.elapsed_secs / 100

        return out.reshape(N, -1), elapsed_time

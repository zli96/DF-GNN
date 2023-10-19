import pdb
import time

import dgl.sparse as dglsp
import torch
import torch.nn as nn

from dgNN.operators.fused_gfconv import GFConvFuse_ELL
from dgNN.utils import benchmark, Timer


class SparseMHA_ELL(nn.Module):
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

    def forward(self, params, h, ell=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, indptr, indices, row_index, rows_per_tb, val = params
        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        if ell:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

            out, elapsed_time = benchmark(
                GFConvFuse_ELL, indptr, indices, row_index, rows_per_tb, val, q, k, v
            )
            out = out.transpose(1, 2)
        else:
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

        return out.reshape(N, -1), elapsed_time * 1000

import pdb
import time

import dgl.sparse as dglsp
import torch
import torch.nn as nn

from dgNN.operators.fused_gfconv import GFConvFuse_hyper, GFConvFuse_hyper_nofuse
from dgNN.utils import benchmark, Timer


class SparseMHA_hyper(nn.Module):
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

    def forward(self, params, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, indptr, indices, rows, val, smem_consume = params
        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        elapsed_time = 0
        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GFConvFuse_hyper, indptr, indices, rows, val, smem_consume, q, k, v
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


class SparseMHA_hyper_nofuse(nn.Module):
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

    def forward(self, params, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, indptr, indices, rows, val, smem_consume = params
        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        elapsed_time = 0
        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GFConvFuse_hyper_nofuse,
                indptr,
                indices,
                rows,
                val,
                smem_consume,
                q,
                k,
                v,
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

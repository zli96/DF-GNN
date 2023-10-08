import pdb
import time

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgNN.operators.fused_gfconv import (
    GFConvFuse,
    GFConvFuse_ELL,
    GFConvFuse_hyper,
    GFConvFuse_outdegree,
    GFConvFuse_subgraph,
)
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

    def forward(self, params, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, indptr, indices, val, smem_consume = params
        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            # print("fuse kernel start")
            # print(f"q {q.shape}, k {k.shape}, v {v.shape}")
            # print(f"indptr {indptr.shape} indices {indices.shape} val {val.shape}")
            # pdb.set_trace()
            # torch.cuda.synchronize()
            # start = time.time()

            out, elapsed_time = benchmark(
                GFConvFuse, indptr, indices, val, smem_consume, q, k, v
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
            # print("------------(q@k)*A-------------")
            # print(A.row)
            # print(A.col)
            # csr_val = [attn.val[i] for i in val_idx]
            # torch.cuda.synchronize()
            elapsed_time = t.elapsed_secs / 100
            # for i in range(5):
            #     for head in range(self.num_heads):
            #         cols = indices[indptr[i]:indptr[i+1]]
            #         vals = torch.tensor([torch.dot(q.transpose(1,2)[i][head],k.transpose(1,2)[col][head]).item() for col in cols])
            #         vals_softmax = torch.softmax(vals, dim=0)
            #         print(f"node {i} head {head}")
            #         print("val ", vals)
            #         print("softmax ",vals_softmax)
            #         print([elem[head].item() for elem in csr_val[indptr[i]:indptr[i+1]]])
            #     print("output", out[i])
            # pdb.set_trace()

        # return self.out_proj(out.reshape(N, -1))
        return out.reshape(N, -1), elapsed_time * 1000


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


class SparseMHA_subgraph(nn.Module):
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
        A, indptr, indices, nodes_subgraph, val = params
        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        elapsed_time = 0
        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GFConvFuse_subgraph, nodes_subgraph, indptr, indices, val, q, k, v
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


class SparseMHA_outdegree(nn.Module):
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
        (
            A,
            row_ptr,
            col_ind,
            val,
            nodes_subgraph,
            smem_nodes_subgraph,
            store_node,
            store_flag,
        ) = params
        ######################################################################
        # (HIGHLIGHT) Compute the multi-head attention with Sparse Matrix API
        ######################################################################
        elapsed_time = 0
        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GFConvFuse_outdegree,
                row_ptr,
                col_ind,
                val,
                nodes_subgraph,
                smem_nodes_subgraph,
                store_node,
                store_flag,
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

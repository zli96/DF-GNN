import pdb
import time
import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..operators.fused_gfconv import GFConvFuse


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
        row_ptr, col_ind, val_idx = A.csr()
        row_ptr = row_ptr.int()
        col_ind = col_ind.int()
        val = torch.tensor([A.val[i] for i in val_idx]).float()
        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            # print("fuse kernel start")
            # print(f"q {q.shape}, k {k.shape}, v {v.shape}")
            # print(f"row_ptr {row_ptr.shape} col_ind {col_ind.shape} val {val.shape}")
            # pdb.set_trace()
            torch.cuda.synchronize()
            start = time.time()
            out = GFConvFuse(row_ptr, col_ind, val, q, k, v)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start
            out = out.transpose(1,2)
            # print("fuse kernel end")
            # print(out.shape)
            # for i in range(5):
            #     print("output", out[i].flatten())
            # pdb.set_trace()
        else:
            torch.cuda.synchronize()
            start = time.time()
            attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # [N, N, nh]
            attn = attn.softmax()
            # print("------------(q@k)*A-------------")
            # print(A.row)
            # print(A.col)
            # csr_val = [attn.val[i] for i in val_idx]
            out = dglsp.bspmm(attn, v)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start
            # for i in range(5):
            #     for head in range(self.num_heads):
            #         cols = col_ind[row_ptr[i]:row_ptr[i+1]]
            #         vals = torch.tensor([torch.dot(q.transpose(1,2)[i][head],k.transpose(1,2)[col][head]).item() for col in cols])
            #         vals_softmax = torch.softmax(vals, dim=0)
            #         print(f"node {i} head {head}")
            #         print("val ", vals)
            #         print("softmax ",vals_softmax)
            #         print([elem[head].item() for elem in csr_val[row_ptr[i]:row_ptr[i+1]]])
            #     print("output", out[i])
            # pdb.set_trace()

        # return self.out_proj(out.reshape(N, -1))
        return out.reshape(N, -1), elapsed_time * 1000

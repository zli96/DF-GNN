import dgl.sparse as dglsp
import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
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
        col_ind =col_ind.int()
        val = torch.tensor([A.val[i]for i in val_idx]).float()
        if fuse:
            # TODO transpose or reshape?
            q = q.transpose(1,2)
            k = k.transpose(1,2)
            v = v.transpose(1,2)
            # q = q.reshape(N, self.num_heads, self.head_dim)
            # k = k.reshape(N, self.num_heads, self.head_dim)
            # v = v.reshape(N, self.num_heads, self.head_dim)
            print("fuse kernel start")
            print(f"q {q.shape}, k {k.shape}, v {v.shape}")
            print(f"row_ptr {row_ptr.shape} col_ind {col_ind.shape} val {val.shape}")
            pdb.set_trace()
            out = GFConvFuse(row_ptr, col_ind, val, q, k, v)
            print("fuse kernel end")
            pdb.set_trace()
        else:
            attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # [N, N, nh]
            print("------------(q@k)*A-------------")
            print(A.row)
            print(A.col)
            csr_val = [attn.val[i] for i in val_idx]
            
            for i in range(5):
                for head in range(self.num_heads):
                    cols = col_ind[row_ptr[i]:row_ptr[i+1]]
                    vals = [torch.dot(q.transpose(1,2)[i][head],k.transpose(1,2)[col][head]).item() for col in cols]
                    print(f"node {i} head {head}")
                    print(vals)
                    print([elem[head].item() for elem in csr_val[row_ptr[i]:row_ptr[i+1]]])
            pdb.set_trace()
            attn = attn.softmax()
            out = dglsp.bspmm(attn, v)

        # return self.out_proj(out.reshape(N, -1))
        return out.reshape(N, -1)

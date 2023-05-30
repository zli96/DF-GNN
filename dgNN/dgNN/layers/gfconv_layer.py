import dgl.sparse as dglsp
import torch.nn as nn
import torch.nn.functional as F
import pdb
from ..operators.fused_gfconv import GFConvFuse

class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=80, num_heads=8):
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
        if fuse:
            row_ptr, col_ind, val = A.csr()
            print("q shape", q.transpose(1,2).shape)
            out = GFConvFuse(row_ptr.int(), col_ind.int(), val.float(), q.transpose(1,2), k.transpose(1,2), v.transpose(1,2))
        else:
            attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # [N, N, nh]
            attn = attn.softmax()
            out = dglsp.bspmm(attn, v)

        return self.out_proj(out.reshape(N, -1))
    

import dgl.sparse as dglsp
import torch.nn as nn


class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, in_size, out_size, num_heads):
        super().__init__()
        self.in_size = in_size
        self.num_heads = num_heads
        self.head_dim = out_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(in_size, out_size)
        self.k_proj = nn.Linear(in_size, out_size)
        self.v_proj = nn.Linear(in_size, out_size)

    def prep_qkv(self, h):
        N = len(h)

        ## get Q, K, V features
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        return q, k, v

    def forward_dglsp(self, A, q, k, v):
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # [N, N, nh]
        attn = attn.softmax()
        out = dglsp.bspmm(attn, v)
        return out

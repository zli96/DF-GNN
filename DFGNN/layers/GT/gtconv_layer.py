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

        self.in_proj = nn.Linear(in_size, out_size)
        self.q_proj = nn.Linear(out_size, out_size)
        self.k_proj = nn.Linear(out_size, out_size)
        self.v_proj = nn.Linear(out_size, out_size)

    def forward_nofuse(self, A, q, k, v):
        # TODO: spmm 要不要用csr格式（但好像更慢）
        # A.csr()
        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # [N, N, nh]
        attn = attn.softmax()
        out = dglsp.bspmm(attn, v)
        return out

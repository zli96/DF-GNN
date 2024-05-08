import dgl.sparse as dglsp
from torch import nn
from torch.nn import functional as F


class AGNNConvDGL(nn.Module):
    def __init__(self, in_size, out_size, num_heads):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.proj = nn.Linear(in_size, out_size)

    def forward_dglsp(self, A, H):
        H_norm = F.normalize(H, p=2, dim=1)
        attn = dglsp.bsddmm(A, H_norm, H_norm.transpose(1, 0))  # [N, N, nh]
        attn = attn.softmax()
        out = dglsp.bspmm(attn, H)
        return out

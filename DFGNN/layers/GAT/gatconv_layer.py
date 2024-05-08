import dgl.sparse as dglsp
import torch
from torch import nn


class GATConvDGL(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout=0, negative_slope=0.2):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_size, out_size * num_heads)
        self.a_l = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.a_r = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)

    ###########################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement
    # multihead attention.
    ###########################################################################
    def forward_dglsp(self, A_hat, Z):
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[A_hat.row] + e_r[A_hat.col]
        a = self.activation(e)
        A_atten = dglsp.val_like(A_hat, a).softmax()
        out = dglsp.bspmm(A_atten, Z)
        return out

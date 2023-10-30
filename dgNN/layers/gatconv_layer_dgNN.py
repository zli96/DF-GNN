import dgl.sparse as dglsp
import torch
from torch import nn

from dgNN.operators.fused_gatconv import GATConvFuse_inference
from dgNN.utils import Timer


class GATConvDGL(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout, negative_slope):
        super().__init__()

        self.out_size = out_size
        self.num_heads = num_heads

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
    def forward(self, A_hat, Z):
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[A_hat.row] + e_r[A_hat.col]
        a = self.activation(e)
        A_atten = dglsp.val_like(A_hat, a).softmax()
        return dglsp.bspmm(A_atten, Z)


class GATConv_dgNN(nn.Module):  # our gat layer
    def __init__(
        self,
        hidden_size,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        bias=True,
    ):
        super(GATConv_dgNN, self).__init__()
        self.in_feats = hidden_size
        self.out_feats = hidden_size
        self.num_heads = num_heads
        self.negative_slope = negative_slope

        self.gat_nofuse = GATConvDGL(
            in_size=self.in_feats,
            out_size=self.out_feats,
            num_heads=self.num_heads,
            dropout=0,
            negative_slope=self.negative_slope,
        )

        self.W = self.gat_nofuse.W
        self.attn_l = self.gat_nofuse.a_l
        self.attn_r = self.gat_nofuse.a_r
        self.attn_drop = attn_drop
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        g, row_ptr, col_ind = params
        if fuse:
            for i in range(3):
                # shape [num_nodes, num_heads, out_feats]
                h = self.W(feat).view(-1, self.out_feats, self.num_heads)
                # shape [num_nodes, num_heads]
                attn_row = (self.attn_l * h).sum(dim=1)
                attn_col = (self.attn_r * h).sum(dim=1)
                h = h.view(-1, self.num_heads, self.out_feats)

                out = GATConvFuse_inference(
                    attn_row, attn_col, row_ptr, col_ind, self.negative_slope, h
                )
            with Timer() as t:
                for i in range(100):
                    # shape [num_nodes, num_heads, out_feats]
                    h = self.W(feat).view(-1, self.out_feats, self.num_heads)
                    # shape [num_nodes, num_heads]
                    attn_row = (self.attn_l * h).sum(dim=1)
                    attn_col = (self.attn_r * h).sum(dim=1)
                    h = h.view(-1, self.num_heads, self.out_feats)
                    out = GATConvFuse_inference(
                        attn_row, attn_col, row_ptr, col_ind, self.negative_slope, h
                    )
        else:
            for i in range(3):

                out = self.gat_nofuse(g, feat)

            with Timer() as t:
                for i in range(100):
                    out = self.gat_nofuse(g, feat)

        elapsed_time = t.elapsed_secs / 100
        return out.reshape(N, -1), elapsed_time * 1000

import dgl.sparse as dglsp

from DFGNN.utils import benchmark

from .gatconv_layer import GATConvDGL


class GATConv_hybrid(GATConvDGL):
    def conv(self, A_hat, A_csr, Z):
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)
        e = e_l[A_hat.row] + e_r[A_hat.col]
        a = self.activation(e)
        dglsp.val_like(A_hat, a).softmax()
        out = dglsp.bspmm(A_csr, Z)
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        A, A2 = params
        A2.csr()
        if fuse:
            out, elapsed_time = benchmark(self.conv, A, A2, feat)
        else:
            out, elapsed_time = benchmark(self.forward_nofuse, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000

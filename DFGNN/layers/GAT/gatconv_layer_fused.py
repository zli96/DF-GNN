from DFGNN.operators.fused_gatconv import (
    GATConvFuse_inference,
    GATConvFuse_inference_hyper,
    GATConvFuse_inference_hyper_recompute,
    GATConvFuse_inference_hyper_v2,
    GATConvFuse_inference_softmax,
)
from DFGNN.utils import benchmark

from .gatconv_layer import GATConvDGL

# our gat layer in hyper method
class GATConv_dgNN(GATConvDGL):
    def conv(self, row_ptr, col_ind, a_l, a_r, h):
        attn_row = (a_l * h).sum(dim=-1)
        attn_col = (a_r * h).sum(dim=-1)
        out = GATConvFuse_inference(
            attn_row, attn_col, row_ptr, col_ind, self.negative_slope, h
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        if fuse:
            row_ptr, col_ind, _, _ = params
            feat = self.W(feat).view(-1, self.num_heads, self.out_size)
            out, elapsed_time = benchmark(
                self.conv,
                row_ptr,
                col_ind,
                self.a_l.transpose(1, 2),
                self.a_r.transpose(1, 2),
                feat,
            )
        else:
            A = params
            feat = self.W(feat).view(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000


class GATConv_hyper(GATConvDGL):
    def conv(self, indptr, indices, rows, smem_consume, a_l, a_r, h):
        attn_row = (a_l * h).sum(dim=-1)
        attn_col = (a_r * h).sum(dim=-1)
        h = h.view(-1, self.num_heads, self.out_size)
        out = GATConvFuse_inference_hyper(
            smem_consume,
            attn_row,
            attn_col,
            indptr,
            indices,
            rows,
            self.negative_slope,
            h,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        if fuse:
            indptr, indices, rows, _, smem_consume = params
            feat = self.W(feat).view(-1, self.num_heads, self.out_size)
            feat = feat.detach().contiguous()
            out, elapsed_time = benchmark(
                self.conv,
                indptr,
                indices,
                rows,
                smem_consume,
                self.a_l.transpose(1, 2),
                self.a_r.transpose(1, 2),
                feat,
            )
        else:
            A = params
            feat = self.W(feat).view(-1, self.out_size, self.num_heads)
            feat = feat.detach().contiguous()
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000


class GATConv_hyper_recompute(GATConvDGL):
    def conv(self, indptr, indices, a_l, a_r, h):
        attn_row = (a_l * h).sum(dim=-1)
        attn_col = (a_r * h).sum(dim=-1)
        out = GATConvFuse_inference_hyper_recompute(
            attn_row,
            attn_col,
            indptr,
            indices,
            self.negative_slope,
            h,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        if fuse:
            indptr, indices, _, _, _ = params
            feat = self.W(feat).view(-1, self.num_heads, self.out_size)
            out, elapsed_time = benchmark(
                self.conv,
                indptr,
                indices,
                self.a_l.transpose(1, 2),
                self.a_r.transpose(1, 2),
                feat,
            )
        else:
            A = params
            feat = self.W(feat).view(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000


class GATConv_softmax(GATConvDGL):
    def conv(self, indptr, indices, rows, smem_consume, a_l, a_r, h):
        attn_row = (a_l * h).sum(dim=-1)
        attn_col = (a_r * h).sum(dim=-1)
        out = GATConvFuse_inference_softmax(
            smem_consume,
            attn_row,
            attn_col,
            indptr,
            indices,
            rows,
            self.negative_slope,
            h,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        if fuse:
            indptr, indices, rows, _, smem_consume = params
            feat = self.W(feat).view(-1, self.num_heads, self.out_size)
            out, elapsed_time = benchmark(
                self.conv,
                indptr,
                indices,
                rows,
                smem_consume,
                self.a_l.transpose(1, 2),
                self.a_r.transpose(1, 2),
                feat,
            )
        else:
            A = params
            feat = self.W(feat).view(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000


class GATConv_hyper_v2(GATConvDGL):
    def conv(self, indptr, indices, smem_consume, a_l, a_r, h):
        out = GATConvFuse_inference_hyper_v2(
            smem_consume,
            a_l,
            a_r,
            indptr,
            indices,
            self.negative_slope,
            h,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        if fuse:
            indptr, indices, _, _, smem_consume = params
            feat = self.W(feat).view(-1, self.num_heads, self.out_size)
            out, elapsed_time = benchmark(
                self.conv,
                indptr,
                indices,
                smem_consume,
                self.a_l.transpose(1, 2),
                self.a_r.transpose(1, 2),
                feat,
            )
        else:
            A = params
            feat = self.W(feat).view(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000

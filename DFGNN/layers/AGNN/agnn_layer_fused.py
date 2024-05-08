from torch.nn import functional as F

from DFGNN.operators.fused_gtconv import (
    GTConvFuse_inference_csr,
    GTConvFuse_inference_hyper,
    GTConvFuse_inference_softmax,
)
from DFGNN.utils import benchmark

from .agnn_layer import AGNNConvDGL


class AGNNConv_csr(AGNNConvDGL):
    def conv(self, H, indptr, indices, val, smem_consume):
        H_norm = F.normalize(H, p=2, dim=-1)
        out = GTConvFuse_inference_csr(
            indptr,
            indices,
            val,
            smem_consume,
            H_norm,
            H_norm,
            H,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        H = self.proj(feat).view(-1, self.num_heads, self.out_size)
        if fuse:
            indptr, indices, val, smem_consume = params
            H = H.contiguous()
            out, elapsed_time = benchmark(
                self.conv,
                H,
                indptr,
                indices,
                val,
                smem_consume,
            )
        else:
            A = params
            H = H.reshape(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, H)
            out = out.transpose(1, 2)
        return out.reshape(N, -1), elapsed_time * 1000


class AGNNConv_softmax(AGNNConvDGL):
    def conv(self, H, indptr, indices, rows, val, smem_consume):
        H_norm = F.normalize(H, p=2, dim=-1)
        out = GTConvFuse_inference_softmax(
            indptr,
            indices,
            rows,
            val,
            smem_consume,
            H_norm,
            H_norm,
            H,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        H = self.proj(feat).view(-1, self.num_heads, self.out_size)
        if fuse:
            indptr, indices, rows, val, smem_consume = params
            H = H.contiguous()
            out, elapsed_time = benchmark(
                self.conv,
                H,
                indptr,
                indices,
                rows,
                val,
                smem_consume,
            )
        else:
            A = params
            H = H.reshape(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, H)
            out = out.transpose(1, 2)
        return out.reshape(N, -1), elapsed_time * 1000


class AGNNConv_hyper(AGNNConvDGL):
    def conv(self, H, indptr, indices, rows, val, smem_consume):
        H_norm = F.normalize(H, p=2, dim=-1)
        out = GTConvFuse_inference_hyper(
            indptr,
            indices,
            rows,
            val,
            smem_consume,
            H_norm,
            H_norm,
            H,
        )
        return out

    def forward(self, params, feat, fuse=False):
        N = len(feat)
        H = self.proj(feat).view(-1, self.num_heads, self.out_size)
        if fuse:
            indptr, indices, rows, val, smem_consume = params
            H = H.contiguous()
            out, elapsed_time = benchmark(
                self.conv,
                H,
                indptr,
                indices,
                rows,
                val,
                smem_consume,
            )
        else:
            A = params
            H = H.reshape(-1, self.out_size, self.num_heads)
            out, elapsed_time = benchmark(self.forward_dglsp, A, H)
            out = out.transpose(1, 2)
        return out.reshape(N, -1), elapsed_time * 1000

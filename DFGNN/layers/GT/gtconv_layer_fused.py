from DFGNN.operators.fused_gtconv import (
    GTConvFuse_inference_csr,
    GTConvFuse_inference_hyper,
    GTConvFuse_inference_softmax,
)
from DFGNN.utils import benchmark

from .gtconv_layer import SparseMHA


class SparseMHA_hyper(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q, k, v = self.prep_qkv(h)

        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            indptr, indices, rows, val, smem_consume = params
            out, elapsed_time = benchmark(
                GTConvFuse_inference_hyper,
                indptr,
                indices,
                rows,
                val,
                smem_consume,
                q,
                k,
                v,
            )
            out = out.transpose(1, 2)
        else:
            A = params
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000


class SparseMHA_CSR(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q, k, v = self.prep_qkv(h)

        if fuse:
            indptr, indices, val, smem_consume = params
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_csr, indptr, indices, val, smem_consume, q, k, v
            )
            out = out.transpose(1, 2)
        else:
            A = params
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000


class SparseMHA_softmax(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q, k, v = self.prep_qkv(h)

        if fuse:
            indptr, indices, rows, val, smem_consume = params
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_softmax,
                indptr,
                indices,
                rows,
                val,
                smem_consume,
                q,
                k,
                v,
            )
            out = out.transpose(1, 2)
        else:
            A = params
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000

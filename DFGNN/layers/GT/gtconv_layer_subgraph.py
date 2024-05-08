from DFGNN.operators.fused_gtconv import (
    GTConvFuse_inference_indegree,
    GTConvFuse_inference_indegree_hyper,
    GTConvFuse_inference_subgraph,
)
from DFGNN.utils import benchmark

from .gtconv_layer import SparseMHA


class SparseMHA_subgraph(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        h = self.in_proj(h)

        ## get Q, K, V features
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        A, indptr, indices, nodes_subgraph, val = params

        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_subgraph,
                nodes_subgraph,
                indptr,
                indices,
                val,
                q,
                k,
                v,
            )
            out = out.transpose(1, 2)
        else:
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000


class SparseMHA_indegree(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        (
            A,
            row_ptr,
            col_ind,
            val,
            nodes_subgraph,
            smem_nodes_subgraph,
            store_node,
            store_flag,
        ) = params

        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_indegree,
                row_ptr,
                col_ind,
                val,
                nodes_subgraph,
                smem_nodes_subgraph,
                store_node,
                store_flag,
                q,
                k,
                v,
            )
            out = out.transpose(1, 2)
        else:
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000


class SparseMHA_indegree_hyper(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)
        (
            A,
            row,
            row_ptr,
            col_ind,
            val,
            nodes_subgraph,
            smem_nodes_subgraph,
            store_node,
            store_flag,
        ) = params

        if fuse:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark(
                GTConvFuse_inference_indegree_hyper,
                row,
                row_ptr,
                col_ind,
                val,
                nodes_subgraph,
                smem_nodes_subgraph,
                store_node,
                store_flag,
                q,
                k,
                v,
            )
            out = out.transpose(1, 2)
        else:
            out, elapsed_time = benchmark(self.forward_dglsp, A, q, k, v)

        return out.reshape(N, -1), elapsed_time * 1000

import torch

from DFGNN.operators.fused_gtconv import GTConvFuse_inference_hyper_ablation
from DFGNN.utils import benchmark
from .gtconv_layer import SparseMHA


def benchmark_flush(function, *args):
    steps = 100

    for i in range(3):
        out = function(*args)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        # flush_cache()
        torch.cuda._sleep(1_000_000)

        start_events[i].record()
        out = function(*args)
        end_events[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return out, sum(times) / steps / 1000


class SparseMHA_hyper_ablation(SparseMHA):
    def forward(self, params, h, fuse=False):
        N = len(h)

        ## get Q, K, V features
        q = self.q_proj(h).reshape(N, self.head_dim, self.num_heads)
        q *= self.scaling
        k = self.k_proj(h).reshape(N, self.head_dim, self.num_heads)
        v = self.v_proj(h).reshape(N, self.head_dim, self.num_heads)

        if fuse:
            indptr, indices, rows, val, smem_consume = params
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out, elapsed_time = benchmark_flush(
                GTConvFuse_inference_hyper_ablation,
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

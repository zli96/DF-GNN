import torch

from dgNN.operators.fused_gatconv import GATConvFuse_inference_hyper_ablation
from dgNN.utils import benchmark
from .gatconv_layer import GATConvDGL


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


class GATConv_hyper_ablation(GATConvDGL):
    def conv(self, indptr, indices, rows, smem_consume, feat):
        h = self.W(feat).view(-1, self.out_size, self.num_heads)
        attn_row = (self.a_l * h).sum(dim=1)
        attn_col = (self.a_r * h).sum(dim=1)
        h = h.view(-1, self.num_heads, self.out_size)
        out = GATConvFuse_inference_hyper_ablation(
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
        A, indptr, indices, rows, _, smem_consume = params
        if fuse:
            out, elapsed_time = benchmark(
                self.conv, indptr, indices, rows, smem_consume, feat
            )
        else:
            out, elapsed_time = benchmark(self.forward_nofuse, A, feat)

        return out.reshape(N, -1), elapsed_time * 1000
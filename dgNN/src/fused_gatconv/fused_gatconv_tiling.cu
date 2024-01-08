#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType, int colSize>
__global__ void fused_gat_tiling(int h, int f, const DType *attn_row,
                                 const DType *attn_col, const int *indptr,
                                 const int *indices, const DType *in_feat,
                                 const DType negative_slope, DType *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim
  const int numThreadsInBlock = blockDim.x * blockDim.y * blockDim.z;

  const int lb = indptr[rid]; // row rid elements
  const int num_neighbor = indptr[rid + 1] - lb;
  static __shared__ DType neigh_nodes_weight[colSize];
  const int hf = h * f;
  const int hfid = hid * f + fid;
  DType acc = 0, partial_sum = 0;
  DType weightMax_old = -INFINITY, weightMax = -INFINITY;

  int loop_neighbor = (num_neighbor + colSize - 1) / colSize;

  for (int j = 0; j < loop_neighbor; j++) {
    const int *indicesoff = indices + lb + j * colSize;
    for (int i = 0; i < colSize / numThreadsInBlock; i++) {
      int eid = i * numThreadsInBlock + fid;
      if (eid + j * colSize < num_neighbor) {
        int dst = __ldg(indicesoff + eid);
        DType weight = attn_row[rid * h + hid] + attn_col[dst * h + hid];
        weight = LeakyRelu(weight, negative_slope);
        neigh_nodes_weight[eid] = weight;
      }
    }
    __syncthreads();

    for (int i = 0; i < colSize / WARP_SIZE; i++) {
      DType weight = -INFINITY;
      int pid = threadIdx.x + (i << 5);
      if (pid < num_neighbor) {
        weight = neigh_nodes_weight[pid];
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
      }
      __syncwarp();
      weightMax = MAX(weight, weightMax);
    }
    __syncthreads();

    DType expweightMax =
        (weightMax_old == weightMax) ? 1.0f : exp(weightMax_old - weightMax);
    partial_sum = partial_sum * expweightMax;
    acc = acc * expweightMax;
    for (int i = 0; i < colSize; i++) {
      if (j * colSize + i < num_neighbor) {
        int cid = indicesoff[i];
        DType weight = neigh_nodes_weight[i];
        DType expweight = exp(weight - weightMax);
        acc += expweight * in_feat[cid * hf + hfid];
        partial_sum += expweight;
      }
    }
    __syncthreads();
    weightMax_old = weightMax;
  }
  partial_sum = (partial_sum != 0) ? 1 / partial_sum : 0;
  out_feat[rid * hf + hfid] = acc * partial_sum;
}

torch::Tensor
gat_tiling_inference_cuda(torch::Tensor attn_row, torch::Tensor attn_col,
                          torch::Tensor indptr, torch::Tensor indices,
                          float negative_slope, torch::Tensor in_feat) {
  const auto m = indptr.size(0) - 1;
  const auto nnz = indices.size(0);
  const auto h = attn_row.size(1);
  const auto f = in_feat.size(2);
  auto devid = attn_row.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);

  const int ntx = WARP_SIZE;
  const int nty = (f + WARP_SIZE - 1) / WARP_SIZE;

  const dim3 nblks(m, h);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL((fused_gat_tiling<float, 128>), nblks, nthrs, 0, h, f,
                   attn_row.data_ptr<float>(), attn_col.data_ptr<float>(),
                   indptr.data_ptr<int>(), indices.data_ptr<int>(),
                   in_feat.data_ptr<float>(), negative_slope,
                   out_feat.data_ptr<float>());
  return out_feat;
}

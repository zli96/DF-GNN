#include "../sddmm/sddmm.cuh"
#include "../util/computeUtil.h"
#include "../util/helper_math.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

template <typename DType>
__global__ void
softMax_SPMM_global_memory(const int h, const int f, const int *indptr,
                           const int *indices, const DType *in_feat,
                           const DType *neigh_nodes_weight, DType *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  DType weightMax = -1e38;
  const int hf = h * f;
  const int hfid = hid * f + fid;
  const DType *neigh_nodes_weight_off = neigh_nodes_weight + lb;

  int loop = (num_neighbor + WARP_SIZE - 1) / WARP_SIZE;
  for (int j = 0; j < loop; j++) {
    DType weight = -1e38;
    int pid = threadIdx.x + (j << 5);
    if (pid < num_neighbor) {
      weight = neigh_nodes_weight_off[pid];
      // weight = 1;
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1) {
      weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
    }
    // warpMax = warpReduceMax(weight);
    __syncwarp();
    weightMax = MAX(weight, weightMax);
  }
  __syncthreads();

  // compute the sum of exp
  DType expAll = 0;
  for (int j = 0; j < loop; j++) {
    int pid = threadIdx.x + (j << 5); // node need to process in loop j
    DType exptmp = 0;
    if (pid < num_neighbor) {
      DType weight = neigh_nodes_weight_off[pid];
      // DType weight = 1;
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1) {
      exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
    }
    __syncwarp();
    expAll += exptmp;
  }
  // handle the node with no neighbor
  expAll = (expAll != 0) ? 1.0f / expAll : 0;

  // compute the output
  DType acc = 0;
  for (int j = 0; j < num_neighbor; j++) {
    int cid = indices[lb + j];
    DType weight = neigh_nodes_weight_off[j];
    // DType weight = 1;
    DType attn_val = exp(weight - weightMax);
    if (fid < f) {
      acc += attn_val * in_feat[cid * hf + hfid];
    }
  }
  if (fid < f)
    // handle the node with no neighbor
    out_feat[rid * hf + hfid] = acc * expAll;
}

void gt_softmax_gm_inference_launch(int m, int nnz, int h, int f,
                                    const int *indptr, const int *indices,
                                    const int *rows, const float *val,
                                    const float *Q, const float *K,
                                    const float *V, float *attn_edge,
                                    float *out_feat) {
  const int ntx = 32; // on feature dimension
  const int nty = 8;  // on out dimension
  const int nbx = (nnz + nty - 1) / nty;
  const int nby = FindNumBlocks<'y'>(h);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL((sddmmCooKernel<float>), nblks, nthrs, 0, f * h, f * h, h,
                   nnz, f, rows, indices, val, Q, K, attn_edge);

  const dim3 nblks2(m, h, 1);
  const dim3 nthrs2(32, (f + 31) / 32, 1);
  // CUDA_KERNEL_CALL((sddmmCsrKernel<float>), nblks2, nthrs2,
  //                  0, h, f, indptr, indices, val, Q, K, attn_edge);
  CUDA_KERNEL_CALL((softMax_SPMM_global_memory<float>), nblks2, nthrs2, 0, h, f,
                   indptr, indices, V, attn_edge, out_feat);
}

torch::Tensor gt_softmax_gm_inference_cuda(torch::Tensor indptr,
                                           torch::Tensor indices,
                                           torch::Tensor rows,
                                           torch::Tensor val, torch::Tensor Q,
                                           torch::Tensor K, torch::Tensor V) {
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto nnz = indices.size(0);  // num of edges
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);
  auto attn_edge = torch::zeros({nnz * h}, options);
  gt_softmax_gm_inference_launch(
      m, nnz, h, f, indptr.data_ptr<int>(), indices.data_ptr<int>(),
      rows.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
      K.data_ptr<float>(), V.data_ptr<float>(), attn_edge.data_ptr<float>(),
      out_feat.data_ptr<float>());
  return out_feat;
}
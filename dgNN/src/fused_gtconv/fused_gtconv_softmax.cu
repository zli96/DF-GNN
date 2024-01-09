#include "../sddmm/sddmm.cuh"
#include "../spmm/spmm.cuh"
#include "../util/computeUtil.h"
#include "../util/helper_math.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

void gt_softmax_inference_launch(int m, int nnz, int h, int f, int smem_consume,
                                 const int *indptr, const int *indices,
                                 const int *rows, const float *val,
                                 const float *Q, const float *K, const float *V,
                                 float *attn_edge, float *out_feat) {
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
  //   CUDA_KERNEL_CALL((sddmmCsrKernel<float>), nblks2, nthrs2,
  //                    0, h, f, indptr, indices, val, Q, K, attn_edge);
  CUDA_KERNEL_CALL((softMax_SPMM<float>), nblks2, nthrs2,
                   (smem_consume) * sizeof(float), h, f, indptr, indices, V,
                   attn_edge, out_feat);
}

std::vector<torch::Tensor>
gt_softmax_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                          torch::Tensor rows, torch::Tensor val,
                          int smem_consume, torch::Tensor Q, torch::Tensor K,
                          torch::Tensor V) {
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto nnz = indices.size(0);  // num of edges
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);
  auto attn_edge = torch::zeros({nnz * h}, options);
  gt_softmax_inference_launch(
      m, nnz, h, f, smem_consume, indptr.data_ptr<int>(),
      indices.data_ptr<int>(), rows.data_ptr<int>(), val.data_ptr<float>(),
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      attn_edge.data_ptr<float>(), out_feat.data_ptr<float>());
  return {out_feat};
}
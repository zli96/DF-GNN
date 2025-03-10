#include "../sddmm/sddmm.cuh"
#include "../spmm/spmm.cuh"
#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>

void gat_softmax_gm_inference_launch(int m, int nnz, int h, int f,
                                     const float *attn_row,
                                     const float *attn_col, const int *indptr,
                                     const int *indices, const int *rows,
                                     float negative_slope, float *attn_edge,
                                     const float *in_feat, float *out_feat) {
  const int ntx = 32;
  const int nty = 8;

  const int nbx = (nnz + nty - 1) / nty;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL((gat_sddmmCooKernel<float>), nblks, nthrs, 0, h, h, h, nnz,
                   negative_slope, rows, indices, attn_row, attn_col,
                   attn_edge);

  const dim3 nblks2(m, h, 1);
  const dim3 nthrs2(32, (f + 31) / 32, 1);
  CUDA_KERNEL_CALL((softMax_SPMM_global_memory<float>), nblks2, nthrs2, 0, h, f,
                   indptr, indices, in_feat, attn_edge, out_feat);
}

torch::Tensor
gat_softmax_gm_inference_cuda(torch::Tensor attn_row, torch::Tensor attn_col,
                              torch::Tensor indptr, torch::Tensor indices,
                              torch::Tensor rows, float negative_slope,
                              torch::Tensor in_feat) {
  const auto m = indptr.size(0) - 1;
  const auto nnz = indices.size(0);
  const auto h = attn_row.size(1);
  const auto f = in_feat.size(2);
  auto devid = attn_row.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::empty({m, h, f}, options);
  auto attn_edge = torch::zeros({nnz * h}, options);

  gat_softmax_gm_inference_launch(
      m, nnz, h, f, attn_row.data_ptr<float>(), attn_col.data_ptr<float>(),
      indptr.data_ptr<int>(), indices.data_ptr<int>(), rows.data_ptr<int>(),
      negative_slope, attn_edge.data_ptr<float>(), in_feat.data_ptr<float>(),
      out_feat.data_ptr<float>());

  return out_feat;
}

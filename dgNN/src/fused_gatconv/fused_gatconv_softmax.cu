#include "../spmm/spmm.cuh"
#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>

template <typename DType>
__global__ void gat_sddmmCooKernel(const int lhs_len, const int rhs_len,
                                   const int out_len, const int nnz,
                                   const float negative_slope, const int *row,
                                   const int *col, const DType *lhs,
                                   const DType *rhs, DType *out) {
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  // process each nnz by one warp 32 threads
  if (ty < nnz) {
    const int src = __ldg(row + ty);
    const int dst = __ldg(col + ty);
    const int eid = ty;

    const DType *lhsoff = lhs + src * lhs_len;
    const DType *rhsoff = rhs + dst * rhs_len;
    DType *outoff = out + eid * out_len;

    // the output feature
    int tx = threadIdx.x; // tx < 32
    if (tx == 0) {
      DType val = lhsoff[0] + rhsoff[0];
      val = LeakyRelu(val, negative_slope);
      outoff[0] = val;
    }
  }
}

void gat_softmax_inference_launch(int m, int nnz, int h, int f,
                                  int smem_consume, const float *attn_row,
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
  const int smem_size = smem_consume * sizeof(float);

  CUDA_KERNEL_CALL((gat_sddmmCooKernel<float>), nblks, nthrs, 0, h, h, h, nnz,
                   negative_slope, rows, indices, attn_row, attn_col,
                   attn_edge);

  const dim3 nblks2(m, h, 1);
  const dim3 nthrs2(32, (f + 31) / 32, 1);
  CUDA_KERNEL_CALL((softMax_SPMM<float>), nblks2, nthrs2,
                   (smem_consume) * sizeof(float), h, f, indptr, indices,
                   in_feat, attn_edge, out_feat);
}

torch::Tensor
gat_softmax_inference_cuda(int smem_consume, torch::Tensor attn_row,
                           torch::Tensor attn_col, torch::Tensor indptr,
                           torch::Tensor indices, torch::Tensor rows,
                           float negative_slope, torch::Tensor in_feat) {
  const auto m = indptr.size(0) - 1;
  const auto nnz = indices.size(0);
  const auto h = attn_row.size(1);
  const auto f = in_feat.size(2);
  auto devid = attn_row.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::empty({m, h, f}, options);
  auto attn_edge = torch::zeros({nnz * h}, options);

  gat_softmax_inference_launch(
      m, nnz, h, f, smem_consume, attn_row.data_ptr<float>(),
      attn_col.data_ptr<float>(), indptr.data_ptr<int>(),
      indices.data_ptr<int>(), rows.data_ptr<int>(), negative_slope,
      attn_edge.data_ptr<float>(), in_feat.data_ptr<float>(),
      out_feat.data_ptr<float>());

  return out_feat;
}

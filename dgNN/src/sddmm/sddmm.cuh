#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

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

template <typename DType>
__global__ void sddmmCooKernel(const int lhs_len, const int rhs_len,
                               const int out_len, const int nnz,
                               const int reduce_size, const int *row,
                               const int *col, const DType *data,
                               const DType *lhs, const DType *rhs, DType *out) {
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  // process each nnz by one warp 32 threads
  if (ty < nnz) {
    const int src = __ldg(row + ty);
    const int dst = __ldg(col + ty);
    const int eid = ty;
    // the Q feature of row node
    const DType *lhsoff = lhs + src * lhs_len;
    // the K feature of col node
    const DType *rhsoff = rhs + dst * rhs_len;
    DType *outoff = out + eid * out_len;
    const DType *dataoff = data + eid * out_len;
    // the output feature
    int tx = threadIdx.x; // tx < 32
    for (int i = blockIdx.y; i < out_len;
         i += gridDim.y) { // over output feature dimension
      DType val = 0;
      for (int j = tx; j < reduce_size; j += 64) {
        val += lhsoff[i * reduce_size + j] * rhsoff[i * reduce_size + j];
        if (j + 32 < reduce_size)
          val += lhsoff[i * reduce_size + j + 32] *
                 rhsoff[i * reduce_size + j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(full_mask, val, offset);
      if (tx == 0) {
        outoff[i] = val * dataoff[i];
      }
    }
  }
}
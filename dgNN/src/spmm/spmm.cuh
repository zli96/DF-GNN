#include "../util/computeUtil.h"
#include "../util/helper_math.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

template <typename DType>
__global__ void softMax_SPMM(const int h, const int f, const int *indptr,
                             const int *indices, const DType *in_feat,
                             const DType *attn_edge, DType *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem;
  DType weightMax = -1e38;
  const int hf = h * f;
  const int hfid = hid * f + fid;

  // init smem
  int loop = (num_neighbor + f - 1) / f;
  for (int j = 0; j < loop; j++) {
    int pid = fid + j * f;
    if (pid < num_neighbor) {
      neigh_nodes_weight[pid] = attn_edge[lb + pid];
    }
  }
  __syncthreads();

  loop = (num_neighbor + WARP_SIZE - 1) / WARP_SIZE;
  for (int j = 0; j < loop; j++) {
    DType weight = -1e38;
    int pid = threadIdx.x + (j << 5);
    if (pid < num_neighbor) {
      weight = neigh_nodes_weight[pid];
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1) {
      weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
    }
    // warpMax = warpReduceMax(weight);
    __syncwarp();
    weightMax = MAX(weight, weightMax);
  }
  // compute the sum of exp
  DType expAll = 0;
  for (int j = 0; j < loop; j++) {
    int pid = threadIdx.x + (j << 5); // node need to process in loop j
    DType exptmp = 0;
    if (pid < num_neighbor) {
      DType weight = neigh_nodes_weight[pid];
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1) {
      exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
    }
    __syncwarp();
    expAll += exptmp;
  }

  // compute the output
  DType acc = 0;
  for (int j = 0; j < num_neighbor; j++) {
    int cid = indices[lb + j];
    DType weight = neigh_nodes_weight[j];
    DType attn_val = exp(weight - weightMax);
    if (fid < f) {
      acc += attn_val * in_feat[cid * hf + hfid];
    }
  }
  if (fid < f)
    // handle the node with no neighbor
    out_feat[rid * hf + hfid] = (expAll != 0) ? acc / expAll : 0;
}

template <typename DType>
__global__ void softMax_SPMM_tiling(const int m, const int nnz, const int h,
                                    const int f, const int *indptr,
                                    const int *indices, const DType *val,
                                    const DType *in_feat,
                                    const DType *attn_edge, DType *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem;
  const int hf = h * f;
  const int hfid = hid * f + fid;

  DType acc = 0, partial_sum = 0;
  DType weightMax_old = -1e38, weightMax = -1e38;
  DType expweight, expweightMax;

  for (int j = 0; j < num_neighbor; j++) {
    DType weight = 0;
    int cid = indices[lb + j];

    if (fid == 0) {
      neigh_nodes_weight[j] = attn_edge[lb + j] * val[lb + j];
    }
    __syncthreads();
    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
    expweight = exp(weight - weightMax);
    expweightMax =
        (weightMax_old == weightMax) ? 1 : exp(weightMax_old - weightMax);
    if (fid < f)
      acc = acc * expweightMax + expweight * in_feat[cid * hf + hfid];
    partial_sum = partial_sum * expweightMax + expweight;
    weightMax_old = weightMax;
  }
  if (fid < f)
    out_feat[rid * hf + hfid] = (partial_sum != 0) ? acc / partial_sum : 0;
}
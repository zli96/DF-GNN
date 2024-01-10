#include "../util/computeUtil.h"
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

  // handle the node with no neighbor
  expAll = (expAll != 0) ? 1.0f / expAll : 0;

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
    out_feat[rid * hf + hfid] = acc * expAll;
}

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
      DType weight = neigh_nodes_weight_off[pid];
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
    DType attn_val = exp(weight - weightMax);
    if (fid < f) {
      acc += attn_val * in_feat[cid * hf + hfid];
    }
  }
  if (fid < f)
    // handle the node with no neighbor
    out_feat[rid * hf + hfid] = acc * expAll;
}

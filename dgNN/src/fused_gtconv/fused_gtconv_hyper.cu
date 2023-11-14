#include "../util/computeUtil.h"
#include "../util/helper_math.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType>
__global__ void fused_gt_hyper(const int m, const int nnz, const int h,
                               const int f, const int *row, const int *indptr,
                               const int *indices, const DType *val,
                               const DType *Q, const DType *K, const DType *V,
                               DType *attn_edge, DType *out_feat) {
  // launch dim (32, 8) * (num_nodes/8, 1)

  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockSize, m);

  // the edge bound of this block
  const int blk_edge_lb = indptr[blk_node_lb];
  const int blk_edge_hb = indptr[blk_node_hb];

  // the num of edges in this block
  const int blk_num_edge = blk_edge_hb - blk_edge_lb;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  // SDDMM, edge parallel
  int nnz_per_warp = (blk_num_edge + blockSize - 1) / blockSize;

  const int *rowoff = row + blk_edge_lb;
  const int *indicesoff = indices + blk_edge_lb;
  const DType *valoff = val + blk_edge_lb;

  int src;
  int dst;
  for (int i = 0; i < nnz_per_warp; i++) {
    int curr_edge = tidy * nnz_per_warp + i;
    // edge bound for curr block
    if (curr_edge < blk_num_edge) {
      src = __ldg(rowoff + curr_edge);
      dst = __ldg(indicesoff + curr_edge);

      // // the Q feature of row node
      const DType *Qoff = Q + src * f * h + hid * f;
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;

      DType att_val = 0;
      for (int j = tidx; j < f; j += 64) {
        // float2 Q2 = reinterpret_cast<const float2*>(Qoff)[j];
        // float2 K2 = reinterpret_cast<const float2*>(Koff)[j];
        // att_val += vecDot2<float2, float>(Q2, K2);
        att_val += Qoff[j] * Koff[j];
        if (j + 32 < f)
          att_val += Qoff[j + 32] * Koff[j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        // TODO consider to move val into smem
        neigh_nodes_weight[curr_edge] = att_val * valoff[curr_edge];
      }
    }
  }
  __syncthreads();

  // Softmax+SPMM, node parallel
  int curr_node = blk_node_lb + tidy;
  if (curr_node < blk_node_hb) {
    const int edge_lb = indptr[curr_node];
    const int edge_hb = indptr[curr_node + 1];
    const int num_edge = edge_hb - edge_lb;

    DType weightMax = -1e38;
    const int hf = h * f;
    // const int hfid = hid * f + tidx;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -1e38;
      int pid = tidx + (j << 5);
      if (pid < num_edge) {
        weight = neigh_nodes_weight_off[pid];
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
      }
      __syncwarp();
      weightMax = MAX(weight, weightMax);
    }

    // if (tidx == 0) {
    //   edge_max[curr_node * h + hid] = weightMax;
    // }

    // compute the sum of exp
    DType expAll = 0;
    for (int j = 0; j < loop; j++) {
      int pid = tidx + (j << 5); // node need to process in loop j
      DType exptmp = 0;
      if (pid < num_edge) {
        DType weight = neigh_nodes_weight_off[pid];
        exptmp = exp(weight - weightMax);
        neigh_nodes_weight_off[pid] = exptmp;
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }

    // if (tidx == 0) {
    //   edge_sum[curr_node * h + hid] = expAll;
    // }

    // compute the output
    int loop_f = (f + WARP_SIZE - 1) / WARP_SIZE;
    for (int i = 0; i < loop_f; i += 1) {
      DType acc = 0;
      int pid = tidx + (i << 5);
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType attn_val = neigh_nodes_weight_off[j];
        if (i == 0 && tidx == 0) {
          attn_edge[hid * nnz + edge_lb + j] = attn_val / expAll;
        }
        if (pid < f)
          acc += attn_val * V[cid * hf + hid * f + pid];
      }
      // handle the node with no neighbor
      if (pid < f)
        out_feat[curr_node * hf + hid * f + pid] =
            (expAll != 0) ? acc / expAll : 0;
    }
  }
}

template <typename DType>
__global__ void fused_gt_hyper_inference(const int m, const int h, const int f,
                                         const int *row, const int *indptr,
                                         const int *indices, const DType *val,
                                         const DType *Q, const DType *K,
                                         const DType *V, DType *out_feat) {
  // launch dim (32, 8) * (num_nodes/8, 1)

  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockSize, m);

  // the edge bound of this block
  const int blk_edge_lb = indptr[blk_node_lb];
  const int blk_edge_hb = indptr[blk_node_hb];

  // the num of edges in this block
  const int blk_num_edge = blk_edge_hb - blk_edge_lb;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  // SDDMM, edge parallel
  int nnz_per_warp = (blk_num_edge + blockSize - 1) / blockSize;

  const int *rowoff = row + blk_edge_lb;
  const int *indicesoff = indices + blk_edge_lb;
  const DType *valoff = val + blk_edge_lb;

  int src;
  int dst;
  for (int i = 0; i < nnz_per_warp; i++) {
    int curr_edge = tidy * nnz_per_warp + i;
    // edge bound for curr block
    if (curr_edge < blk_num_edge) {
      src = __ldg(rowoff + curr_edge);
      dst = __ldg(indicesoff + curr_edge);

      // // the Q feature of row node
      const DType *Qoff = Q + src * f * h + hid * f;
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;

      DType att_val = 0;
      for (int j = tidx; j < f; j += 64) {
        // float2 Q2 = reinterpret_cast<const float2*>(Qoff)[j];
        // float2 K2 = reinterpret_cast<const float2*>(Koff)[j];
        // att_val += vecDot2<float2, float>(Q2, K2);
        att_val += Qoff[j] * Koff[j];
        if (j + 32 < f)
          att_val += Qoff[j + 32] * Koff[j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        // TODO consider to move val into smem
        neigh_nodes_weight[curr_edge] = att_val * valoff[curr_edge];
      }
    }
  }
  __syncthreads();

  // Softmax+SPMM, node parallel
  int curr_node = blk_node_lb + tidy;
  if (curr_node < blk_node_hb) {
    const int edge_lb = indptr[curr_node];
    const int edge_hb = indptr[curr_node + 1];
    const int num_edge = edge_hb - edge_lb;

    DType weightMax = -1e38;
    const int hf = h * f;
    // const int hfid = hid * f + tidx;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -1e38;
      int pid = tidx + (j << 5);
      if (pid < num_edge) {
        weight = neigh_nodes_weight_off[pid];
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
      }
      __syncwarp();
      weightMax = MAX(weight, weightMax);
    }

    // compute the sum of exp
    DType expAll = 0;
    for (int j = 0; j < loop; j++) {
      int pid = tidx + (j << 5); // node need to process in loop j
      DType exptmp = 0;
      if (pid < num_edge) {
        DType weight = neigh_nodes_weight_off[pid];
        exptmp = exp(weight - weightMax);
        neigh_nodes_weight_off[pid] = exptmp;
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }

    // compute the output
    int loop_f = (f + WARP_SIZE - 1) / WARP_SIZE;
    // int loop_f = (f + WARP_SIZE* 2 - 1) / WARP_SIZE/2;
    // float2 * out_feat2 = reinterpret_cast<float2*>(out_feat);

    // for (int i = 0; i < loop_f; i++) {
    //   float2 acc=make_float2(0,0);
    //   int pid = tidx + 64 * i;
    //   for (int j = 0; j < num_edge; j++) {
    //     int cid = indices[edge_lb + j];
    //     DType attn_val = neigh_nodes_weight_off[j];
    //     if (pid < f)
    //     {
    //       float2 V2 = reinterpret_cast<const float2*>(V)[(cid * hf + hid *
    //       f)/2 + pid]; acc += attn_val * V2;
    //     }

    //   }
    //   // handle the node with no neighbor
    //   if (pid < f)
    //     out_feat2[(curr_node * hf + hid * f)/2 + pid] =
    //         (expAll != 0) ? acc / expAll : make_float2(0,0);
    // }

    for (int i = 0; i < loop_f; i += 1) {
      DType acc = 0;
      int pid = tidx + (i << 5);
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType attn_val = neigh_nodes_weight_off[j];
        if (pid < f)
          acc += attn_val * V[cid * hf + hid * f + pid];
      }
      // handle the node with no neighbor
      if (pid < f)
        out_feat[curr_node * hf + hid * f + pid] =
            (expAll != 0) ? acc / expAll : 0;
    }
  }
}

template <typename DType>
__global__ void fused_inference_kernel_hyper_row_switch(
    const int m, const int h, const int f, const int *row, const int *indptr,
    const int *indices, const DType *val, const DType *Q, const DType *K,
    const DType *V, DType *out_feat) {
  // launch dim (32, 8) * (num_nodes/8, 1)
  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockSize, m);

  // the edge bound of this block
  const int blk_edge_lb = indptr[blk_node_lb];
  const int blk_edge_hb = indptr[blk_node_hb];

  // the num of edges in this block
  const int blk_num_edge = blk_edge_hb - blk_edge_lb;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  float Q_row[32];

  // SDDMM, edge parallel
  int nnz_per_warp = (blk_num_edge + blockSize - 1) / blockSize;

  const int *rowoff = row + blk_edge_lb;
  const int *indicesoff = indices + blk_edge_lb;
  const DType *valoff = val + blk_edge_lb;
  // DType *Q_smemoff = Q_smem + tidy * f;

  int src_old = -1;
  int src;
  int dst;
  for (int i = 0; i < nnz_per_warp; i++) {
    int curr_edge = tidy * nnz_per_warp + i;
    // edge bound for curr block
    if (curr_edge < blk_num_edge) {
      src = __ldg(rowoff + curr_edge);
      dst = __ldg(indicesoff + curr_edge);
      if (src != src_old) {
        src_old = src;
        for (int j = tidx; j < f; j += 64) {
          int pid = j / WARP_SIZE;
          Q_row[pid] = Q[src_old * f * h + hid * f + j];
          if (j + 32 < f) {
            Q_row[pid + 1] = Q[src_old * f * h + hid * f + j + 32];
          }
        }
      }
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;
      DType att_val = 0;
      for (int j = tidx; j < f; j += 64) {
        int idx = j / WARP_SIZE;
        att_val += Q_row[idx] * Koff[j];
        if (j + 32 < f)
          att_val += Q_row[idx + 1] * Koff[j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        // TODO consider to move val into smem
        neigh_nodes_weight[curr_edge] = att_val * valoff[curr_edge];
      }
    }
  }
  __syncthreads();

  // Softmax+SPMM, node parallel
  int curr_node = blk_node_lb + tidy;
  if (curr_node < blk_node_hb) {
    const int edge_lb = indptr[curr_node];
    const int edge_hb = indptr[curr_node + 1];
    const int num_edge = edge_hb - edge_lb;

    DType weightMax = -1e38;
    const int hf = h * f;
    // const int hfid = hid * f + tidx;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -1e38;
      int pid = tidx + (j << 5);
      if (pid < num_edge) {
        weight = neigh_nodes_weight_off[pid];
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
      }
      __syncwarp();
      weightMax = MAX(weight, weightMax);
    }

    // compute the sum of exp
    DType expAll = 0;
    for (int j = 0; j < loop; j++) {
      int pid = tidx + (j << 5); // node need to process in loop j
      DType exptmp = 0;
      if (pid < num_edge) {
        DType weight = neigh_nodes_weight_off[pid];
        exptmp = exp(weight - weightMax);
        neigh_nodes_weight_off[pid] = exptmp;
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }

    // compute the output
    int loop_f = (f + WARP_SIZE - 1) / WARP_SIZE;
    for (int i = 0; i < loop_f; i++) {
      DType acc = 0;
      int pid = tidx + (i << 5);
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType attn_val = neigh_nodes_weight_off[j];
        if (pid < f)
          acc += attn_val * V[cid * hf + hid * f + pid];
      }
      // handle the node with no neighbor
      if (pid < f)
        out_feat[curr_node * hf + hid * f + pid] =
            (expAll != 0) ? acc / expAll : 0;
    }
  }
}

// SPMM backward A @ B = C -> dB = (A^T) @ dC
template <typename DType>
__global__ void
spmm_backward_kernel(int h, int f, const int *col_ptr, const int *row_ind,
                     const int *val_idx, const DType *Q, const DType *K,
                     const DType *attn_edge, const DType *grad, DType *grad_V) {
  const int cid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature di
  const int lb = col_ptr[cid];
  const int hb = col_ptr[cid + 1];
  const int num_neighbor = hb - lb;

  // compute the output
  DType acc = 0;
  for (int j = 0; j < num_neighbor; j++) {
    int rid = row_ind[lb + j];
    DType weight = attn_edge[val_idx[lb + j]];
    if (fid < f) {
      acc += weight * grad[rid * h * f + hid * f + fid];
    }
  }
  if (fid < f)
    // handle the node with no neighbor
    grad_V[cid * h * f + hid * f + fid] = acc;
}

// SPMM backward A @ B = C -> dA = (dC @ B^T) * A
template <typename DType>
__global__ void fused_backward_kernel(int m, int h, int f, const int *row,
                                      const int *row_ptr, const int *col_ind,
                                      const DType *K, const DType *V,
                                      const DType *attn_edge, const DType *grad,
                                      DType *grad_Q) {
  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockSize, m);

  // the edge bound of this block
  const int blk_edge_lb = row_ptr[blk_node_lb];
  const int blk_edge_hb = row_ptr[blk_node_hb];

  // the num of edges in this block
  const int blk_num_edge = blk_edge_hb - blk_edge_lb;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  int nnz_per_warp = (blk_num_edge + blockSize - 1) / blockSize;

  const DType *lhs = grad;
  const DType *rhs = V;

  const int *rowoff = row + blk_edge_lb;
  const int *indicesoff = col_ind + blk_edge_lb;

  int src;
  int dst;
  // SDDMM, edge parallel
  for (int i = 0; i < nnz_per_warp; i++) {
    int curr_edge = tidy * nnz_per_warp + i;
    // edge bound for curr block
    if (curr_edge < blk_num_edge) {
      src = __ldg(rowoff + curr_edge);
      dst = __ldg(indicesoff + curr_edge);

      const DType *lhsoff = lhs + src * f * h + hid * f;
      const DType *rhsoff = rhs + dst * f * h + hid * f;

      DType att_val = 0;
      for (int j = tidx; j < f; j += 64) {
        att_val += lhsoff[j] * rhsoff[j];
        if (j + 32 < f)
          att_val += lhsoff[j + 32] * rhsoff[j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        neigh_nodes_weight[curr_edge] = att_val;
      }
    }
  }
  __syncthreads();

  int curr_node = blk_node_lb + tidy;
  if (curr_node < blk_node_hb) {
    const int edge_lb = row_ptr[curr_node];
    const int edge_hb = row_ptr[curr_node + 1];
    const int num_edge = edge_hb - edge_lb;

    const int hf = h * f;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    // compute the output
    int loop_f = (f + WARP_SIZE - 1) / WARP_SIZE;

    DType prodsum = 0;
    for (int j = 0; j < loop; j++) {
      int pid = tidx + (j << 5); // node need to process in loop j
      DType prod = 0;
      if (pid < num_edge) {
        DType grad_i = neigh_nodes_weight_off[pid];
        DType out = attn_edge[edge_lb + pid];
        prod = grad_i * out;
        neigh_nodes_weight_off[pid] = prod;
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        prod += __shfl_xor_sync(0xffffffff, prod, stride, 32);
      }
      __syncwarp();
      prodsum += prod;
    }

    for (int i = 0; i < loop_f; i += 1) {
      DType acc = 0;
      int pid = tidx + (i << 5);
      for (int j = 0; j < num_edge; j++) {
        int cid = col_ind[edge_lb + j];
        DType attn_score = attn_edge[edge_lb + j];
        DType attn_val = neigh_nodes_weight_off[j] - prodsum * attn_score;
        if (pid < f)
          acc += attn_val * K[cid * hf + hid * f + pid];
      }
      if (pid < f)
        grad_Q[curr_node * hf + hid * f + pid] = acc;
    }
  }
}

void gt_backward(int m, int n, int nnz, int h, int f, int smem_consume,
                 int *row, int *row_ptr, int *col_ind, float *val, int *col_ptr,
                 int *row_ind, int *val_idx, float *Q, float *K, float *V,
                 float *attn_edge,
                 float *grad,   // input grad
                 float *grad_Q, // output grad
                 float *grad_K, // output grad
                 float *grad_V) // output grad
{
  const dim3 nblks(n, h, 1);
  const dim3 nthrs(32, (f + 31) / 32, 1);
  CUDA_KERNEL_CALL((spmm_backward_kernel<float>), nblks, nthrs,
                   512 * sizeof(float), h, f, col_ptr, row_ind, val_idx, Q, K,
                   attn_edge, grad, grad_V);

  const dim3 nblks2((m + 7) / 8, h, 1);
  const dim3 nthrs2(32, 8, 1);
  const int smem_size = smem_consume * sizeof(float);
  CUDA_KERNEL_CALL((fused_backward_kernel<float>), nblks2, nthrs2, smem_size, m,
                   h, f, row, row_ptr, col_ind, K, V, attn_edge, grad, grad_Q);

  // else
  // {
  //   mhspmm_backward_kernel_small_f<<<dim3(m, 1, 1), dim3(32, h, 1), 32 * h *
  //   sizeof(float)>>>(
  //       m, nnz, h, f, negative_slope, attn_drop, row_ptr, col_ind, col_ptr,
  //       row_ind,permute, edge_max, edge_sum, edge_mask, attn_row, attn_col,
  //       grad, grad_feat);
  // }

  // mhsddmm<<<dim3(nnz / 16 + (nnz & 15), h, 1), dim3(32, 4, 1)>>>(
  //     m, f, h, nnz, row_ptr, col_ind, grad, in_feat, grad_edge_csr);

  // fused_backward_kernel<<<dim3(m, 1, 1), dim3(32, h, 1)>>>(
  //     m, nnz, h, f, attn_drop, row_ptr, col_ind, negative_slope, edge_max,
  //     edge_sum, edge_mask, attn_row, attn_col, grad_edge_csr, grad_attn_row,
  //     grad_attn_col);

  // gather_col<<<dim3(m, 1, 1), dim3(32, h, 1)>>>(m, nnz, h, f, col_ptr,
  // permute, grad_edge_for_gather_csr, grad_attn_col);
}

std::vector<torch::Tensor>
gt_hyper_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                        torch::Tensor rows, torch::Tensor val, int smem_consume,
                        torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto nnz = indices.size(0);  // num of edges
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);

  const int ntx = 32;
  const int nty = 8;

  const int nbx = (m + nty - 1) / nty;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int smem_size = smem_consume * sizeof(float);

  CUDA_KERNEL_CALL((fused_gt_hyper_inference<float>), nblks, nthrs, smem_size,
                   m, h, f, rows.data_ptr<int>(), indptr.data_ptr<int>(),
                   indices.data_ptr<int>(), val.data_ptr<float>(),
                   Q.data_ptr<float>(), K.data_ptr<float>(),
                   V.data_ptr<float>(), out_feat.data_ptr<float>());

  return {out_feat};
}

std::vector<torch::Tensor>
gt_hyper_forward_cuda(torch::Tensor row_ptr, torch::Tensor col_ind,
                      torch::Tensor rows, torch::Tensor val,
                      torch::Tensor col_ptr, torch::Tensor row_ind,
                      torch::Tensor val_idx, int smem_consume, torch::Tensor Q,
                      torch::Tensor K, torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto m = row_ptr.size(0) - 1; // num of nodes
  const auto nnz = col_ind.size(0);   // num of edges
  const auto h = Q.size(1);           // num of heads
  const auto f = Q.size(2);           // num of feats
  auto devid = row_ptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);
  auto attn_edge = torch::empty({h, nnz}, options);

  const int ntx = 32;
  const int nty = 8;

  const int nbx = (m + nty - 1) / nty;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int smem_size = smem_consume * sizeof(float);

  CUDA_KERNEL_CALL((fused_gt_hyper<float>), nblks, nthrs, smem_size, m, nnz, h,
                   f, rows.data_ptr<int>(), row_ptr.data_ptr<int>(),
                   col_ind.data_ptr<int>(), val.data_ptr<float>(),
                   Q.data_ptr<float>(), K.data_ptr<float>(),
                   V.data_ptr<float>(), attn_edge.data_ptr<float>(),
                   out_feat.data_ptr<float>());

  return {out_feat, attn_edge};
}

std::vector<torch::Tensor>
gt_backward_cuda(torch::Tensor row_ptr, torch::Tensor col_ind,
                 torch::Tensor rows, torch::Tensor val, torch::Tensor col_ptr,
                 torch::Tensor row_ind, torch::Tensor val_idx, int smem_consume,
                 torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                 torch::Tensor attn_edge, torch::Tensor grad) {

  const auto m = row_ptr.size(0) - 1;
  const auto n = col_ptr.size(0) - 1;
  printf("m %d n %d \n", m, n);
  const auto nnz = col_ind.size(0);
  const auto h = Q.size(1);
  const auto f = Q.size(2);
  auto devid = row_ptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);

  // auto grad_edge_csr = torch::empty({nnz, h}, options);
  auto grad_Q = torch::empty({m, h, f}, options);
  auto grad_K = torch::empty({n, h, f}, options);
  auto grad_V = torch::empty({n, h, f}, options);

  gt_backward(m, n, nnz, h, f, smem_consume, rows.data_ptr<int>(),
              row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(),
              val.data_ptr<float>(), col_ptr.data_ptr<int>(),
              row_ind.data_ptr<int>(), val_idx.data_ptr<int>(),
              Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
              attn_edge.data_ptr<float>(), grad.data_ptr<float>(),
              grad_Q.data_ptr<float>(), grad_K.data_ptr<float>(),
              grad_V.data_ptr<float>());

  return {grad_Q, grad_K, grad_V};
}
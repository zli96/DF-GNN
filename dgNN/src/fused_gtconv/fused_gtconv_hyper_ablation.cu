#include "../util/computeUtil.h"
#include "../util/helper_math.h"
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <torch/types.h>
using namespace std;

// GT conv without warp-balanced SDDMM && redundency-free softmax
template <typename DType>
__global__ void fused_gt_hyper_inference_no_optimization(
    const int m, const int h, const int f, const int *row, const int *indptr,
    const int *indices, const DType *val, const DType *Q, const DType *K,
    const DType *V, DType *out_feat) {
  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_edge_lb = indptr[blk_node_lb];

  int curr_node = blk_node_lb + tidy;
  if (curr_node < m) {
    const int *rowoff = row + blk_edge_lb;
    const int *indicesoff = indices + blk_edge_lb;
    const DType *valoff = val + blk_edge_lb;

    const int edge_lb = __ldg(indptr + curr_node);
    const int num_edge = __ldg(indptr + curr_node + 1) - edge_lb;

    for (int i = 0; i < num_edge; i++) {
      int curr_edge = edge_lb + i - blk_edge_lb;
      // edge bound for curr block
      int src = __ldg(rowoff + curr_edge);
      int dst = __ldg(indicesoff + curr_edge);

      // // the Q feature of row node
      const DType *Qoff = Q + src * f * h + hid * f;
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;

      DType att_val = 0;
      // for (int j = tidx; j < f / 4; j += 32) {
      //   float4 Q2 = reinterpret_cast<const float4 *>(Qoff)[j];
      //   float4 K2 = reinterpret_cast<const float4 *>(Koff)[j];
      //   att_val += vecDot4<float4, float>(Q2, K2);
      // }
      for (int j = tidx; j < f; j += 32) {
        att_val += Qoff[j] * Koff[j];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        neigh_nodes_weight[curr_edge] = att_val * valoff[curr_edge];
      }
    }

    // TODO why this reduce latency?
    __syncthreads();

    DType weightMax = -INFINITY;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -INFINITY;
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
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }
    expAll = (expAll != 0) ? 1.0f / expAll : 0;
    // compute the output
    for (int i = tidx; i < f; i += 32) {
      DType acc = 0;
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType weight = neigh_nodes_weight_off[j];
        DType attn_val = exp(weight - weightMax);
        acc += attn_val * V[cid * h * f + hid * f + i];
      }
      // handle the node with no neighbor
      out_feat[curr_node * h * f + hid * f + i] = acc * expAll;
    }
  }
}

template <typename DType>
__global__ void fused_gt_hyper_inference_balanced_SDDMM(
    const int m, const int h, const int f, const int *row, const int *indptr,
    const int *indices, const DType *val, const DType *Q, const DType *K,
    const DType *V, DType *out_feat) {
  // launch dim (32, 8) * (num_nodes/8, 1)

  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockSize, m);
  const int blk_edge_lb = indptr[blk_node_lb];

  // the num of edges in this block
  const int blk_num_edge = indptr[blk_node_hb] - blk_edge_lb;

  int nnz_per_warp = (blk_num_edge + blockSize - 1) / blockSize;

  const int *rowoff = row + blk_edge_lb;
  const int *indicesoff = indices + blk_edge_lb;
  const DType *valoff = val + blk_edge_lb;

  // SDDMM, edge parallel
  for (int i = 0; i < nnz_per_warp; i++) {
    int curr_edge = tidy * nnz_per_warp + i;
    // edge bound for curr block
    if (curr_edge < blk_num_edge) {
      int src = __ldg(rowoff + curr_edge);
      int dst = __ldg(indicesoff + curr_edge);

      // the Q feature of row node
      const DType *Qoff = Q + src * f * h + hid * f;
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;
      DType att_val = 0;
      // for (int j = tidx; j < f / 4; j += 32) {
      //   float4 Q2 = reinterpret_cast<const float4 *>(Qoff)[j];
      //   float4 K2 = reinterpret_cast<const float4 *>(Koff)[j];
      //   att_val += vecDot4<float4, float>(Q2, K2);
      // }
      for (int j = tidx; j < f; j += 32) {
        att_val += Qoff[j] * Koff[j];
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
  if (curr_node < m) {
    const int edge_lb = __ldg(indptr + curr_node);
    const int num_edge = __ldg(indptr + curr_node + 1) - edge_lb;

    DType weightMax = -INFINITY;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -INFINITY;
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
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }
    expAll = (expAll != 0) ? 1.0f / expAll : 0;

    // compute the output
    for (int i = tidx; i < f; i += 32) {
      DType acc = 0;
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType weight = neigh_nodes_weight_off[j];
        DType attn_val = exp(weight - weightMax);
        acc += attn_val * V[cid * h * f + hid * f + i];
      }
      // handle the node with no neighbor
      out_feat[curr_node * h * f + hid * f + i] = acc * expAll;
    }
    // DType *Outoff = out_feat + curr_node * h * f + hid * f;
    // for (int i = tidx; i < f / 4; i += 32) {
    //   // DType acc = 0;
    //   DType acc[4] = {0, 0, 0, 0};
    //   for (int j = 0; j < num_edge; j++) {
    //     int cid = indices[edge_lb + j];
    //     DType weight = neigh_nodes_weight_off[j];
    //     DType attn_val = exp(weight - weightMax);
    //     const DType *Voff = V + cid * h * f + hid * f + 4 * i;
    //     Mul4_const<float>(acc, Voff, attn_val);
    //   }
    //   // handle the node with no neighbor
    //   selfMulConst4<float>(acc, expAll);
    //   Store<float4, float>(Outoff, acc, 4 * i);
    // }
  }
}

template <typename DType>
__global__ void fused_gt_hyper_inference_softmax(
    const int m, const int h, const int f, const int *row, const int *indptr,
    const int *indices, const DType *val, const DType *Q, const DType *K,
    const DType *V, DType *out_feat) {
  // launch dim (32, 8) * (num_nodes/8, 1)

  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_edge_lb = indptr[blk_node_lb];

  int curr_node = blk_node_lb + tidy;
  if (curr_node < m) {
    const int *rowoff = row + blk_edge_lb;
    const int *indicesoff = indices + blk_edge_lb;
    const DType *valoff = val + blk_edge_lb;

    const int edge_lb = __ldg(indptr + curr_node);
    const int num_edge = __ldg(indptr + curr_node + 1) - edge_lb;

    for (int i = 0; i < num_edge; i++) {
      int curr_edge = edge_lb + i - blk_edge_lb;
      // edge bound for curr block
      int src = __ldg(rowoff + curr_edge);
      int dst = __ldg(indicesoff + curr_edge);

      // // the Q feature of row node
      const DType *Qoff = Q + src * f * h + hid * f;
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;

      DType att_val = 0;
      // for (int j = tidx; j < f / 4; j += 32) {
      //   float4 Q2 = reinterpret_cast<const float4 *>(Qoff)[j];
      //   float4 K2 = reinterpret_cast<const float4 *>(Koff)[j];
      //   att_val += vecDot4<float4, float>(Q2, K2);
      // }
      for (int j = tidx; j < f; j += 32) {
        att_val += Qoff[j] * Koff[j];
        // if (j + 32 < f)
        //   att_val += Qoff[j + 32] * Koff[j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        neigh_nodes_weight[curr_edge] = att_val * valoff[curr_edge];
      }
    }

    __syncthreads();
    DType weightMax = -INFINITY;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -INFINITY;
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
    expAll = (expAll != 0) ? 1.0f / expAll : 0;

    // compute the output
    for (int i = tidx; i < f; i += 32) {
      DType acc = 0;
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType attn_val = neigh_nodes_weight_off[j];
        acc += attn_val * V[cid * h * f + hid * f + i];
      }
      // handle the node with no neighbor
      out_feat[curr_node * h * f + hid * f + i] = acc * expAll;
    }
    // DType *Outoff = out_feat + curr_node * h * f + hid * f;
    // for (int i = tidx; i < f / 4; i += 32) {
    //   // DType acc = 0;
    //   DType acc[4] = {0, 0, 0, 0};
    //   for (int j = 0; j < num_edge; j++) {
    //     int cid = indices[edge_lb + j];
    //     DType attn_val = neigh_nodes_weight_off[j];
    //     const DType *Voff = V + cid * h * f + hid * f + 4 * i;
    //     Mul4_const<float>(acc, Voff, attn_val);
    //   }
    //   // handle the node with no neighbor
    //   selfMulConst4<float>(acc, expAll);
    //   Store<float4, float>(Outoff, acc, 4 * i);
    // }
  }
}

// GT conv without warp-balanced SDDMM && redundency-free softmax
template <typename DType>
__global__ void fused_gt_hyper_inference_node_parallel(
    const int m, const int h, const int f, const int *row, const int *indptr,
    const int *indices, const DType *val, const DType *Q, const DType *K,
    const DType *V, DType *out_feat) {
  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem; // [8, f]

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_edge_lb = indptr[blk_node_lb];

  int curr_node = blk_node_lb + tidy;
  if (curr_node < m) {
    const int edge_lb = __ldg(indptr + curr_node);
    const int num_edge = __ldg(indptr + curr_node + 1) - edge_lb;

    const int *rowoff = row + edge_lb;
    const int *indicesoff = indices + edge_lb;
    const DType *valoff = val + edge_lb;

    for (int i = 0; i < num_edge; i++) {
      // edge bound for curr block
      int src = __ldg(rowoff + i);
      int dst = __ldg(indicesoff + i);

      // // the Q feature of row node
      const DType *Qoff = Q + src * f * h + hid * f;
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;

      DType att_val = 0;
      for (int j = tidx; j < f; j += 32) {
        att_val += Qoff[j] * Koff[j];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        neigh_nodes_weight[i] = att_val * valoff[i];
      }
    }

    // TODO why this reduce latency?
    __syncthreads();

    DType weightMax = -INFINITY;

    DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -INFINITY;
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
    expAll = (expAll != 0) ? 1.0f / expAll : 0;
    // compute the output
    for (int i = tidx; i < f; i += 32) {
      DType acc = 0;
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType attn_val = neigh_nodes_weight_off[j];
        acc += attn_val * V[cid * h * f + hid * f + i];
      }
      // handle the node with no neighbor
      out_feat[curr_node * h * f + hid * f + i] = acc * expAll;
    }
  }
}

// GT conv with global memory store intermediate value, without edge-parallel
// SDDMM
template <typename DType>
__global__ void fused_gt_hyper_inference_global_memory(
    const int m, const int h, const int f, const int *row, const int *indptr,
    const int *indices, const DType *val, const DType *Q, const DType *K,
    const DType *V, DType *neigh_nodes_weight, DType *out_feat) {
  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;

  int curr_node = blk_node_lb + tidy;
  if (curr_node < m) {
    const int edge_lb = __ldg(indptr + curr_node);
    const int num_edge = __ldg(indptr + curr_node + 1) - edge_lb;

    const int *rowoff = row + edge_lb;
    const int *indicesoff = indices + edge_lb;
    const DType *valoff = val + edge_lb;
    DType *neigh_nodes_weight_off = neigh_nodes_weight + edge_lb;

    for (int i = 0; i < num_edge; i++) {
      // edge bound for curr block
      int src = __ldg(rowoff + i);
      int dst = __ldg(indicesoff + i);

      // // the Q feature of row node
      const DType *Qoff = Q + src * f * h + hid * f;
      // the K feature of col node
      const DType *Koff = K + dst * f * h + hid * f;

      DType att_val = 0;
      for (int j = tidx; j < f; j += 32) {
        att_val += Qoff[j] * Koff[j];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (tidx == 0) {
        neigh_nodes_weight_off[i] = att_val * valoff[i];
      }
    }
    __syncthreads();

    DType weightMax = -INFINITY;

    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -INFINITY;
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
    __syncthreads();

    // compute the sum of exp
    DType expAll = 0;
    for (int j = 0; j < loop; j++) {
      int pid = tidx + (j << 5); // node need to process in loop j
      DType exptmp = 0;
      if (pid < num_edge) {
        DType weight = neigh_nodes_weight_off[pid];
        exptmp = exp(weight - weightMax);
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }
    expAll = (expAll != 0) ? 1.0f / expAll : 0;
    __syncthreads();

    // compute the output
    for (int i = tidx; i < f; i += 32) {
      DType acc = 0;
      for (int j = 0; j < num_edge; j++) {
        int cid = indices[edge_lb + j];
        DType weight = neigh_nodes_weight_off[j];
        DType attn_val = exp(weight - weightMax);
        acc += attn_val * V[cid * h * f + hid * f + i];
      }
      // handle the node with no neighbor
      out_feat[curr_node * h * f + hid * f + i] = acc * expAll;
    }
  }
}

// GT conv without edge parallel-SDDMM
std::vector<torch::Tensor>
gt_hyper_inference_ablation_cuda(torch::Tensor indptr, torch::Tensor indices,
                                 torch::Tensor rows, torch::Tensor val,
                                 int smem_consume, torch::Tensor Q,
                                 torch::Tensor K, torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto nnz = indices.size(0);  // num of edges
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  const auto val_size = val.size(0); // check if val is scalar
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

  char *mode = std::getenv("alblation_mode");
  if (strcmp(mode, "0") == 0) {
    // printf("mode 0\n");
    CUDA_KERNEL_CALL(
        (fused_gt_hyper_inference_no_optimization<float>), nblks, nthrs,
        smem_size, m, h, f, rows.data_ptr<int>(), indptr.data_ptr<int>(),
        indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
        K.data_ptr<float>(), V.data_ptr<float>(), out_feat.data_ptr<float>());
  } else if (strcmp(mode, "1") == 0) {
    CUDA_KERNEL_CALL(
        (fused_gt_hyper_inference_balanced_SDDMM<float>), nblks, nthrs,
        smem_size, m, h, f, rows.data_ptr<int>(), indptr.data_ptr<int>(),
        indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
        K.data_ptr<float>(), V.data_ptr<float>(), out_feat.data_ptr<float>());
  } else if (strcmp(mode, "2") == 0) {
    CUDA_KERNEL_CALL(
        (fused_gt_hyper_inference_softmax<float>), nblks, nthrs, smem_size, m,
        h, f, rows.data_ptr<int>(), indptr.data_ptr<int>(),
        indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
        K.data_ptr<float>(), V.data_ptr<float>(), out_feat.data_ptr<float>());
  } else if (strcmp(mode, "3") == 0) {
    auto neigh_nodes_weight = torch::zeros({nnz}, options);
    CUDA_KERNEL_CALL(
        (fused_gt_hyper_inference_global_memory<float>), nblks, nthrs, 0, m, h,
        f, rows.data_ptr<int>(), indptr.data_ptr<int>(),
        indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
        K.data_ptr<float>(), V.data_ptr<float>(),
        neigh_nodes_weight.data_ptr<float>(), out_feat.data_ptr<float>());
  } else if (strcmp(mode, "4") == 0) {
    CUDA_KERNEL_CALL(
        (fused_gt_hyper_inference_node_parallel<float>), nblks, nthrs,
        smem_size, m, h, f, rows.data_ptr<int>(), indptr.data_ptr<int>(),
        indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
        K.data_ptr<float>(), V.data_ptr<float>(), out_feat.data_ptr<float>());
  }

  return {out_feat};
}
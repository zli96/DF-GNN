#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType>
__global__ void
fused_forward_kernel_hyper(const int blockSize, const int warps_row,
                           const int m, const int h, const int f,
                           const int *row, const int *indptr,
                           const int *indices, const DType *val, const DType *Q,
                           const DType *K, const DType *V, DType *out_feat) {
  // launch dim (32, 8) * (num_nodes/8, 1), blockSize:8, warps_row:1,only
  // support h = 1 launch dim (64, 4) * (num_nodes/4, 1), blockSize:4,
  // warps_row:2,only support h = 1

  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;

  // const int blockSize = blockDim.y;
  // const int warps_row = roundup(f, WARP_SIZE) / WARP_SIZE;
  const int warps_num = blockSize * warps_row;

  // the node bound of this block
  const int blk_node_lb = blockSize * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockSize, m);

  // the edge bound of this block
  const int blk_edge_lb = indptr[blk_node_lb];
  const int blk_edge_hb = indptr[blk_node_hb];

  // the num of edges in this block
  const int blk_num_edge = blk_edge_hb - blk_edge_lb;

  const int laneId = tidx % WARP_SIZE;
  const int warpId = tidx / WARP_SIZE + tidy * warps_row;

  // init smem
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem;

  // SDDMM, edge parallel
  int loop = (blk_num_edge + warps_num - 1) / warps_num;
  const int *rowoff = row + blk_edge_lb;
  const int *indicesoff = indices + blk_edge_lb;
  const DType *valoff = val + blk_edge_lb;

  for (int i = 0; i < loop; i++) {
    int curr_edge = i * warps_num + warpId;
    if (curr_edge < blk_num_edge) {
      const int src = __ldg(rowoff + curr_edge);
      const int dst = __ldg(indicesoff + curr_edge);

      // the Q feature of row node
      const DType *Qoff = Q + src * f * h;
      // the K feature of col node
      const DType *Koff = K + dst * f * h;

      DType att_val = 0;
      for (int j = laneId; j < f; j += 64) {
        att_val += Qoff[hid * f + j] * Koff[hid * f + j];
        if (j + 32 < f)
          att_val += Qoff[hid * f + j + 32] * Koff[hid * f + j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        att_val += __shfl_down_sync(full_mask, att_val, offset);
      if (laneId == 0) {
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
    const int hfid = hid * f + tidx;

    const DType *neigh_nodes_weight_off =
        neigh_nodes_weight + (edge_lb - blk_edge_lb);

    loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -1e38;
      int pid = laneId + (j << 5);
      if (pid < num_edge) {
        weight = neigh_nodes_weight_off[pid];
      }
      __syncwarp();
      for (int stride = 16; stride > 0; stride >>= 1) {
        weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32), weight);
      }
      __syncwarp();
      weightMax = MAX(weight, weightMax);
    }

    // compute the sum of exp
    DType expAll = 0;
    for (int j = 0; j < loop; j++) {
      int pid = laneId + (j << 5); // node need to process in loop j
      DType exptmp = 0;
      if (pid < num_edge) {
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

    // compute the output
    DType acc = 0;
    DType attn_val;
    for (int j = 0; j < num_edge; j++) {
      int cid = indices[edge_lb + j];
      DType weight = neigh_nodes_weight_off[j];
      attn_val = exp(weight - weightMax);
      if (tidx < f)
        acc += attn_val * V[cid * hf + hfid];
    }
    // handle the node with no neighbor
    if (tidx < f)
      out_feat[curr_node * hf + hfid] = (expAll != 0) ? acc / expAll : 0;
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

__global__ void sddmmCsrKernel(const int m, const int nnz, const int h,
                               const int f, const int *indptr,
                               const int *indices, const float *val,
                               const float *Q, const float *K,
                               float *attn_edge) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  static __shared__ float warpLevelSums[WARP_SIZE];
  const int hf = h * f;
  const int hfid = hid * f + fid;
  const int laneId = fid % WARP_SIZE;
  const int warpId = fid / WARP_SIZE;
  const int blockSize = blockDim.x * blockDim.y;

  float Q_i = Q[rid * hf + hfid];

  for (int j = 0; j < num_neighbor; j++) {
    float weight_partial = 0;

    int cid = indices[lb + j];
    weight_partial = Q_i * K[cid * hf + hfid];

    __syncthreads();
    weight_partial = warpReduceSum(weight_partial, blockSize);
    if (laneId == 0)
      warpLevelSums[warpId] = weight_partial;
    __syncthreads();
    weight_partial = (fid < blockSize / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if (warpId == 0)
      weight_partial = warpReduceSum(weight_partial, blockSize / WARP_SIZE);
    if (fid == 0) {
      attn_edge[lb + j] = weight_partial * val[lb + j];
    }
  }
  __syncthreads();
}

__global__ void softMax_SPMM(const int m, const int nnz, const int h,
                             const int f, const int *indptr, const int *indices,
                             const float *val, const float *V,
                             const float *attn_edge, float *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  extern __shared__ float smem[];
  float *neigh_nodes_weight = smem;
  float weightMax = -1e38;
  const int hf = h * f;
  const int hfid = hid * f + fid;

  // for (int j = 0; j < num_neighbor; j++)
  // {
  //   float weight = 0;
  //   if (fid == 0)
  //   {
  //     neigh_nodes_weight[j] = attn_edge[lb + j];
  //   }
  //   __syncthreads();
  //   weight = neigh_nodes_weight[j];
  //   weightMax = MAX(weight, weightMax);
  // }
  // // compute the sum of exp
  // int loop = (num_neighbor + WARP_SIZE - 1) / WARP_SIZE;

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
    float weight = -1e38;
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
  float expAll = 0;
  for (int j = 0; j < loop; j++) {
    int pid = threadIdx.x + (j << 5); // node need to process in loop j
    float exptmp = 0;
    if (pid < num_neighbor) {
      float weight = neigh_nodes_weight[pid];
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
  float acc = 0;
  float attn_val;
  for (int j = 0; j < num_neighbor; j++) {
    int cid = indices[lb + j];
    float weight = neigh_nodes_weight[j];
    attn_val = exp(weight - weightMax);
    if (fid < f) {
      acc += attn_val * V[cid * hf + hfid];
    }
  }
  if (fid < f)
    // handle the node with no neighbor
    out_feat[rid * hf + hfid] = (expAll != 0) ? acc / expAll : 0;
}

__global__ void softMax_SPMM_tiling(const int m, const int nnz, const int h,
                                    const int f, const int *indptr,
                                    const int *indices, const float *val,
                                    const float *V, const float *attn_edge,
                                    float *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  extern __shared__ float smem[];
  float *neigh_nodes_weight = smem;
  const int hf = h * f;
  const int hfid = hid * f + fid;

  float acc = 0, partial_sum = 0;
  float weightMax_old = -1e38, weightMax = -1e38;
  float expweight, expweightMax;

  for (int j = 0; j < num_neighbor; j++) {
    float weight = 0;
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
      acc = acc * expweightMax + expweight * V[cid * hf + hfid];
    partial_sum = partial_sum * expweightMax + expweight;
    weightMax_old = weightMax;
  }
  if (fid < f)
    out_feat[rid * hf + hfid] = (partial_sum != 0) ? acc / partial_sum : 0;
}

void gf_forward_hyper_nofuse(int m, int nnz, int h, int f, int smem_consume,
                             const int *indptr, const int *indices,
                             const int *rows, const float *val, const float *Q,
                             const float *K, const float *V, float *attn_edge,
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
  CUDA_KERNEL_CALL((softMax_SPMM), nblks2, nthrs2,
                   (smem_consume) * sizeof(float), m, nnz, h, f, indptr,
                   indices, val, V, attn_edge, out_feat);
}

void gf_forward_hyper_fuse(int m, int nnz, int h, int f, int smem_consume,
                           const int *indptr, const int *indices,
                           const int *rows, const float *val, const float *Q,
                           const float *K, const float *V, float *out_feat) {
  const int ntx = roundup(f, WARP_SIZE);
  const int nty = 256 / ntx;

  const int nbx = (m + nty - 1) / nty;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int smem_size = smem_consume * sizeof(float);

  CUDA_KERNEL_CALL((fused_forward_kernel_hyper<float>), nblks, nthrs, smem_size,
                   nty, ntx / WARP_SIZE, m, h, f, rows, indptr, indices, val, Q,
                   K, V, out_feat);
}

std::vector<torch::Tensor>
gf_hyper_fused_forward_cuda(torch::Tensor indptr, torch::Tensor indices,
                            torch::Tensor rows, torch::Tensor val,
                            int smem_consume, torch::Tensor Q, torch::Tensor K,
                            torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto nnz = indices.size(0);  // num of edges
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);
  auto attn_edge = torch::zeros({nnz * h}, options);
  // gf_forward_hyper_nofuse(m, nnz, h, f, smem_consume, indptr.data_ptr<int>(),
  //                   indices.data_ptr<int>(), rows.data_ptr<int>(),
  //                   val.data_ptr<float>(), Q.data_ptr<float>(),
  //                   K.data_ptr<float>(), V.data_ptr<float>(),
  //                   attn_edge.data_ptr<float>(), out_feat.data_ptr<float>());
  gf_forward_hyper_fuse(m, nnz, h, f, smem_consume, indptr.data_ptr<int>(),
                        indices.data_ptr<int>(), rows.data_ptr<int>(),
                        val.data_ptr<float>(), Q.data_ptr<float>(),
                        K.data_ptr<float>(), V.data_ptr<float>(),
                        out_feat.data_ptr<float>());

  return {out_feat};
}

std::vector<torch::Tensor>
gf_hyper_nofuse_forward_cuda(torch::Tensor indptr, torch::Tensor indices,
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
  gf_forward_hyper_nofuse(
      m, nnz, h, f, smem_consume, indptr.data_ptr<int>(),
      indices.data_ptr<int>(), rows.data_ptr<int>(), val.data_ptr<float>(),
      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
      attn_edge.data_ptr<float>(), out_feat.data_ptr<float>());
  return {out_feat};
}
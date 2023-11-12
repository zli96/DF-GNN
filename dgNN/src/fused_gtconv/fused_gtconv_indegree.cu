#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType>
__global__ void fused_gt_indegree_hyper(
    const int h, const int f, const int *node_num_ptr,
    const int *smem_nodes_ptr, const int *store_nodes, const int *store_flags,
    const int *row, const int *indptr, const int *indices, const DType *val,
    const DType *Q, const DType *K, const DType *V, DType *out_feat) {

  // grid: batch_size*h  block: dim * (1024/dim) each tb processes one graph
  const int gid = blockIdx.x;   // index of subgraph
  const int hid = blockIdx.y;   // index of head
  const int tidx = threadIdx.x; // index of feature
  const int tidy = threadIdx.y;

  const int BLOCK_SIZE = blockDim.y;

  // Offset of nodes in the subgraph on the full graph
  const int node_lb = node_num_ptr[gid];
  const int node_hb = node_num_ptr[gid + 1];
  const int num_nodes = node_hb - node_lb; // num of nodes in this subgraph

  const int smem_node_lb = smem_nodes_ptr[gid];
  const int smem_node_hb = smem_nodes_ptr[gid + 1];
  // num of nodes in smem of this subgraph
  const int smem_num_nodes = smem_node_hb - smem_node_lb;

  const int f_mul_32 = roundup(f, 32);

  const int hf = h * f;
  // const int hfid = hid * f + fid;
  // const int laneId = fid % WARP_SIZE;
  // const int warpId = fid / WARP_SIZE;

  // init shared memory
  extern __shared__ DType smem[];
  DType *K_SMEM = smem;
  DType *V_SMEM = (DType *)&K_SMEM[smem_num_nodes * f];
  DType *neigh_nodes_weight = (DType *)&V_SMEM[smem_num_nodes * f];

  // Put the K and V into smem
  int loops_smem_node = (smem_num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int j = 0; j < loops_smem_node; j++) {
    int curr_node_local = j * BLOCK_SIZE + tidy;
    for (int i = 0; i < f; i += 32) {
      int pid = i + tidx;
      if (curr_node_local + smem_node_lb < smem_node_hb && i + tidx < f) {
        int curr_node_global = store_nodes[curr_node_local + smem_node_lb];
        K_SMEM[curr_node_local * f + pid] =
            K[curr_node_global * hf + hid * f + pid];
        V_SMEM[curr_node_local * f + pid] =
            V[curr_node_global * hf + hid * f + pid];
      }
    }
  }
  __syncthreads();

  int loops_node = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int j = 0; j < loops_node; j++) {
    // the node bound of this block
    const int blk_node_lb_g = node_lb + j * BLOCK_SIZE;
    const int blk_node_hb_g = node_lb + MIN((j + 1) * BLOCK_SIZE, num_nodes);

    // the edge bound of this block
    const int blk_edge_lb = indptr[blk_node_lb_g];
    const int blk_edge_hb = indptr[blk_node_hb_g];

    // the num of edges in this block
    const int blk_num_edge = blk_edge_hb - blk_edge_lb;

    // SDDMM, edge parallel
    int loop_edge = (blk_num_edge + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // TODO: add row augment
    const int *rowoff = row + blk_edge_lb;
    const int *indicesoff = indices + blk_edge_lb;
    const DType *valoff = val + blk_edge_lb;

    for (int i = 0; i < loop_edge; i++) {
      int curr_edge = i * BLOCK_SIZE + tidy;
      if (curr_edge < blk_num_edge) {
        const int src = __ldg(rowoff + curr_edge);
        const int dst = __ldg(indicesoff + curr_edge);

        // the Q feature of row node
        const DType *Qoff = Q + src * f * h;
        // the K feature of col node
        const DType *Koff = K + dst * f * h;

        DType att_val = 0;
        for (int j = tidx; j < f; j += 64) {
          att_val += Qoff[hid * f + j] * Koff[hid * f + j];
          if (j + 32 < f)
            att_val += Qoff[hid * f + j + 32] * Koff[hid * f + j + 32];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
          att_val += __shfl_down_sync(full_mask, att_val, offset);
        if (tidx == 0) {
          neigh_nodes_weight[curr_edge] = att_val * valoff[curr_edge];
        }
      }
    }
    __syncthreads();
    // Softmax+SPMM, node parallel
    int curr_node = blk_node_lb_g + tidy;
    if (curr_node < blk_node_hb_g) {
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
          int store_flag = store_flags[cid];
          if (pid < f) {
            if (store_flag >= 0)
              acc += attn_val * V_SMEM[store_flag * f + pid];
            else
              acc += attn_val * V[cid * f + hid * f + pid];
          }
        }
        // handle the node with no neighbor
        if (pid < f)
          out_feat[curr_node * hf + hid * f + pid] =
              (expAll != 0) ? acc / expAll : 0;
      }
    }
  }
}

template <typename DType>
__global__ void
fused_gt_indegree(const int h, const int f, const int *node_num_ptr,
                  const int *smem_nodes_ptr, const int *store_nodes,
                  const int *store_flags, const int *indptr, const int *indices,
                  const DType *val, const DType *Q, const DType *K,
                  const DType *V, DType *out_feat) {

  // grid: batch_size*h  block: dim * (1024/dim) each tb processes one graph
  const int gid = blockIdx.x;  // index of subgraph
  const int hid = blockIdx.y;  // index of head
  const int fid = threadIdx.x; // index of feature
  const int tidy = threadIdx.y;

  const int BLOCK_SIZE = blockDim.y;

  // Offset of nodes in the subgraph on the full graph
  const int node_lb = node_num_ptr[gid];
  const int node_hb = node_num_ptr[gid + 1];
  const int num_nodes = node_hb - node_lb; // num of nodes in this subgraph

  const int smem_node_lb = smem_nodes_ptr[gid];
  const int smem_node_hb = smem_nodes_ptr[gid + 1];
  // num of nodes in smem of this subgraph
  const int smem_num_nodes = smem_node_hb - smem_node_lb;

  const int f_mul_32 = roundup(f, 32);

  const int hf = h * f;
  const int hfid = hid * f + fid;
  const int laneId = fid % WARP_SIZE;
  const int warpId = fid / WARP_SIZE;

  // init shared memory
  extern __shared__ DType smem[];
  DType *K_SMEM = smem;
  DType *V_SMEM = (DType *)&K_SMEM[smem_num_nodes * f];
  DType *warpLevelSums = (DType *)&V_SMEM[smem_num_nodes * f];
  DType *neigh_nodes_weight = (DType *)&warpLevelSums[WARP_SIZE * BLOCK_SIZE];

  // Put the K and V into smem
  int loops_smem_node = (smem_num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int j = 0; j < loops_smem_node; j++) {
    int curr_node_local = j * BLOCK_SIZE + tidy;
    if (curr_node_local + smem_node_lb < smem_node_hb && fid < f) {
      int curr_node_global = store_nodes[curr_node_local + smem_node_lb];
      K_SMEM[curr_node_local * f + fid] = K[curr_node_global * hf + hfid];
      V_SMEM[curr_node_local * f + fid] = V[curr_node_global * hf + hfid];
    }
  }
  __syncthreads();

  int loops_node = (num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int j = 0; j < loops_node; j++) {
    int curr_node = tidy + j * BLOCK_SIZE;
    int curr_node_global = curr_node + node_lb;
    if (curr_node_global < node_hb) {
      DType weightMax = -1e38;
      DType Q_i = Q[curr_node_global * hf + hfid]; // Q of current node

      const int lb = indptr[curr_node_global]; // row rid elements
      const int hb = indptr[curr_node_global + 1];
      const int num_neighbor = hb - lb; // num of neighbors

      // SPMM on Q and K
      for (int k = 0; k < num_neighbor; k++) {
        int cid_global = indices[lb + k]; // node id in this subgraph
        DType weight = 0;
        DType weight_partial = 0;
        int store_flag = store_flags[cid_global];
        if (fid < f) {
          if (store_flag >= 0)
            weight_partial = Q_i * K_SMEM[store_flag * f + fid];
          else
            weight_partial = Q_i * K[cid_global * f + fid];
        }
        __syncwarp();
        weight_partial = warpReduceSum(weight_partial, f_mul_32);
        if (laneId == 0) {
          warpLevelSums[WARP_SIZE * tidy + warpId] = weight_partial;
        }
        namedBarrierSync(tidy, f_mul_32);

        weight_partial = (fid < f_mul_32 / WARP_SIZE)
                             ? warpLevelSums[WARP_SIZE * tidy + laneId]
                             : 0;
        if (warpId == 0)
          weight_partial = warpReduceSum(weight_partial, f_mul_32 / WARP_SIZE);
        if (fid == 0) {
          neigh_nodes_weight[((k >> 5) * BLOCK_SIZE + tidy) * 32 + k % 32] =
              weight_partial * val[lb + k];
        }
        namedBarrierSync(tidy, f_mul_32);

        weight =
            neigh_nodes_weight[((k >> 5) * BLOCK_SIZE + tidy) * 32 + k % 32];
        weightMax = MAX(weight, weightMax);
      }

      // Calculate the sum of softmax on attention weight
      int loop_WARP_neigh = (num_neighbor + WARP_SIZE - 1) / WARP_SIZE;
      DType expAll = 0;
      for (int k = 0; k < loop_WARP_neigh; k++) {
        DType exptmp = 0;
        int pid = laneId + (k << 5);
        if (pid < num_neighbor) {
          // TODO need to fix the bank conflict?
          DType weight =
              neigh_nodes_weight[((pid >> 5) * BLOCK_SIZE + tidy) * 32 +
                                 laneId];
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
      for (int k = 0; k < num_neighbor; k++) {
        int cid_global = indices[lb + k];
        DType weight =
            neigh_nodes_weight[((k >> 5) * BLOCK_SIZE + tidy) * 32 + k % 32];
        attn_val = exp(weight - weightMax);
        int store_flag = store_flags[cid_global];
        if (fid < f) {
          if (store_flag >= 0)
            acc += attn_val * V_SMEM[store_flag * f + fid];
          else
            acc += attn_val * V[cid_global * f + fid];
        }
      }
      if (fid < f) {
        // handle the node with no neighbor
        out_feat[curr_node_global * hf + hfid] =
            (expAll != 0) ? acc / expAll : 0;
      }
    }
  }
}

void gt_indegree_inference_launch(int num_subgraph, int h, int f,
                                const int *nodes_subgraph,
                                const int *smem_nodes_subgraph,
                                const int *store_node, const int *store_flag,
                                const int *indptr, const int *indices,
                                const float *val, const float *Q,
                                const float *K, const float *V,
                                float *out_feat) {
  const int ntx = roundup(f, WARP_SIZE);
  const int nty = 1024 / ntx;
  const int nbx = num_subgraph;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int smem_size = 1024 * 64;
  // printf("launch dim %d %d %d %d \n", ntx, nty, nbx, nby);

  cudaFuncSetAttribute(fused_gt_indegree<float>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  CUDA_KERNEL_CALL((fused_gt_indegree<float>), nblks, nthrs, smem_size, h, f,
                   nodes_subgraph, smem_nodes_subgraph, store_node, store_flag,
                   indptr, indices, val, Q, K, V, out_feat);
}

void gt_inference_indegree_hyper(int num_subgraph, int h, int f,
                               const int *nodes_subgraph,
                               const int *smem_nodes_subgraph,
                               const int *store_node, const int *store_flag,
                               const int *row, const int *indptr,
                               const int *indices, const float *val,
                               const float *Q, const float *K, const float *V,
                               float *out_feat) {
  const int ntx = 32;
  const int nty = 32;
  const int nbx = num_subgraph;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int smem_size = 1024 * 64;
  // printf("launch dim %d %d %d %d \n", ntx, nty, nbx, nby);

  cudaFuncSetAttribute(fused_gt_indegree_hyper<float>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  CUDA_KERNEL_CALL((fused_gt_indegree_hyper<float>), nblks, nthrs, smem_size, h,
                   f, nodes_subgraph, smem_nodes_subgraph, store_node,
                   store_flag, row, indptr, indices, val, Q, K, V, out_feat);
}

std::vector<torch::Tensor> gt_indegree_inference_cuda(
    torch::Tensor nodes_subgraph, torch::Tensor smem_nodes_subgraph,
    torch::Tensor store_node, torch::Tensor store_flag, torch::Tensor indptr,
    torch::Tensor indices, torch::Tensor val, torch::Tensor Q, torch::Tensor K,
    torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto num_subgraph = nodes_subgraph.size(0) - 1;
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);

  gt_indegree_inference_launch(
      num_subgraph, h, f, nodes_subgraph.data_ptr<int>(),
      smem_nodes_subgraph.data_ptr<int>(), store_node.data_ptr<int>(),
      store_flag.data_ptr<int>(), indptr.data_ptr<int>(),
      indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
      K.data_ptr<float>(), V.data_ptr<float>(), out_feat.data_ptr<float>());

  return {out_feat};
}

std::vector<torch::Tensor> gt_indegree_hyper_inference_cuda(
    torch::Tensor nodes_subgraph, torch::Tensor smem_nodes_subgraph,
    torch::Tensor store_node, torch::Tensor store_flag, torch::Tensor row,
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor val,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto num_subgraph = nodes_subgraph.size(0) - 1;
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);

  gt_inference_indegree_hyper(
      num_subgraph, h, f, nodes_subgraph.data_ptr<int>(),
      smem_nodes_subgraph.data_ptr<int>(), store_node.data_ptr<int>(),
      store_flag.data_ptr<int>(), row.data_ptr<int>(), indptr.data_ptr<int>(),
      indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
      K.data_ptr<float>(), V.data_ptr<float>(), out_feat.data_ptr<float>());

  return {out_feat};
}
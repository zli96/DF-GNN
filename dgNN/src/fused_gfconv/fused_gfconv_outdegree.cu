#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType, int BLOCK_SIZE, int LOG_BLOCK_SIZE>
__global__ void fused_forward_kernel_outdegree_mul32(
    const int h, const int f, const int *node_num_ptr,
    const int *smem_nodes_ptr, const int *store_nodes, const int *store_flags,
    const int *indptr, const int *indices, const DType *val, const DType *Q,
    const DType *K, const DType *V, DType *out_feat) {
  // grid: 4096*h  block: 32 * 8 each tb processes one graph
  // BLOCK_SIZE = blockDim.y
  const int gid = blockIdx.x;  // index of subgraph
  const int hid = blockIdx.y;  // index of head
  const int fid = threadIdx.x; // index of feature
  const int tidy = threadIdx.y;

  const int node_lb = node_num_ptr[gid];
  const int node_hb =
      node_num_ptr[gid +
                   1]; // Offset of nodes in the subgraph on the full graph
  const int num_nodes = node_hb - node_lb; // num of nodes in this subgraph

  const int smem_node_lb = smem_nodes_ptr[gid];
  const int smem_node_hb = smem_nodes_ptr[gid + 1];
  const int smem_num_nodes =
      node_hb - node_lb; // num of nodes in smem of this subgraph

  const int hf = h * f;
  const int hfid = hid * f + fid;
  const int laneId = fid % WARP_SIZE;
  const int warpId = fid / WARP_SIZE;

  // init shared memory
  static __shared__ DType warpLevelSums[WARP_SIZE * BLOCK_SIZE];
  extern __shared__ DType smem[];
  DType *K_SMEM = smem;
  DType *V_SMEM = (DType *)&K_SMEM[smem_num_nodes * f];
  DType *neigh_nodes_weight = (DType *)&V_SMEM[smem_num_nodes * f];

  // Put the K and V into smem
  int loops_smem_node = (smem_num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int j = 0; j < loops_smem_node; j++) {
    int curr_node_local = j * BLOCK_SIZE + tidy;
    if (curr_node_local + smem_node_lb < smem_node_hb) {
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
        if (store_flag >= 0)
          weight_partial = Q_i * K_SMEM[store_flag * f + fid];
        else
          weight_partial = Q_i * K[cid_global * f + fid];
        __syncwarp();
        weight_partial = warpReduceSum(weight_partial, f);
        if (laneId == 0) {
          warpLevelSums[WARP_SIZE * tidy + warpId] = weight_partial;
        }
        namedBarrierSync(tidy, f);

        weight_partial = (fid < f / WARP_SIZE)
                             ? warpLevelSums[WARP_SIZE * tidy + laneId]
                             : 0;
        if (warpId == 0)
          weight_partial = warpReduceSum(weight_partial, f / WARP_SIZE);
        if (fid == 0) {
          neigh_nodes_weight[tidy + (k << LOG_BLOCK_SIZE)] =
              weight_partial * val[lb + k];
        }
        namedBarrierSync(tidy, f);

        weight = neigh_nodes_weight[tidy + (k << LOG_BLOCK_SIZE)];
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
          DType weight = neigh_nodes_weight[tidy + (pid << LOG_BLOCK_SIZE)];
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
        DType weight = neigh_nodes_weight[tidy + (k << LOG_BLOCK_SIZE)];
        attn_val = exp(weight - weightMax);
        int store_flag = store_flags[cid_global];
        if (store_flag >= 0)
          acc += attn_val * V_SMEM[store_flag * f + fid];
        else
          acc += attn_val * V[cid_global * f + fid];
      }
      // handle the node with no neighbor
      out_feat[curr_node_global * hf + hfid] = (expAll != 0) ? acc / expAll : 0;
    }
  }
}

template <typename DType, int BLOCK_SIZE, int LOG_BLOCK_SIZE>
__global__ void fused_forward_kernel_outdegree(
    const int h, const int f, const int *node_num_ptr,
    const int *smem_nodes_ptr, const int *store_nodes, const int *store_flags,
    const int *indptr, const int *indices, const DType *val, const DType *Q,
    const DType *K, const DType *V, DType *out_feat) {

  // grid: batch_size*h  block: dim * (1024/dim) each tb processes one graph
  const int gid = blockIdx.x;  // index of subgraph
  const int hid = blockIdx.y;  // index of head
  const int fid = threadIdx.x; // index of feature
  const int tidy = threadIdx.y;

  // Offset of nodes in the subgraph on the full graph
  const int node_lb = node_num_ptr[gid];
  const int node_hb = node_num_ptr[gid + 1];
  const int num_nodes = node_hb - node_lb; // num of nodes in this subgraph

  const int smem_node_lb = smem_nodes_ptr[gid];
  const int smem_node_hb = smem_nodes_ptr[gid + 1];
  // num of nodes in smem of this subgraph
  const int smem_num_nodes = node_hb - node_lb;

  const int f_mul_32 = roundup(f, 32);

  const int hf = h * f;
  const int hfid = hid * f + fid;
  const int laneId = fid % WARP_SIZE;
  const int warpId = fid / WARP_SIZE;

  // init shared memory
  static __shared__ DType warpLevelSums[WARP_SIZE * BLOCK_SIZE];
  extern __shared__ DType smem[];
  DType *K_SMEM = smem;
  DType *V_SMEM = (DType *)&K_SMEM[smem_num_nodes * f];
  DType *neigh_nodes_weight = (DType *)&V_SMEM[smem_num_nodes * f];

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
          neigh_nodes_weight[tidy + (k << LOG_BLOCK_SIZE)] =
              weight_partial * val[lb + k];
        }
        namedBarrierSync(tidy, f_mul_32);

        weight = neigh_nodes_weight[tidy + (k << LOG_BLOCK_SIZE)];
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
          DType weight = neigh_nodes_weight[tidy + (pid << LOG_BLOCK_SIZE)];
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
        DType weight = neigh_nodes_weight[tidy + (k << LOG_BLOCK_SIZE)];
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

void gf_forward_outdegree(int num_subgraph, int h, int f,
                          const int *nodes_subgraph,
                          const int *smem_nodes_subgraph, const int *store_node,
                          const int *store_flag, const int *indptr,
                          const int *indices, const float *val, const float *Q,
                          const float *K, const float *V, float *out_feat) {
  const int ntx = roundup(f, WARP_SIZE);
  const int nty = 1024 / ntx;
  const int nbx = num_subgraph;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int smem_size = 1024 * 64 - nty * 32 * 4;
  // printf("launch dim %d %d %d %d \n", ntx, nty, nbx, nby);
  switch (nty) {
  case 8:
    cudaFuncSetAttribute(fused_forward_kernel_outdegree<float, 8, 3>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    CUDA_KERNEL_CALL((fused_forward_kernel_outdegree<float, 8, 3>), nblks,
                     nthrs, smem_size, h, f, nodes_subgraph,
                     smem_nodes_subgraph, store_node, store_flag, indptr,
                     indices, val, Q, K, V, out_feat);
    break;
  case 16:
    cudaFuncSetAttribute(fused_forward_kernel_outdegree<float, 16, 4>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    CUDA_KERNEL_CALL((fused_forward_kernel_outdegree<float, 16, 4>), nblks,
                     nthrs, smem_size, h, f, nodes_subgraph,
                     smem_nodes_subgraph, store_node, store_flag, indptr,
                     indices, val, Q, K, V, out_feat);
    break;
  case 32:
    cudaFuncSetAttribute(fused_forward_kernel_outdegree<float, 32, 5>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    CUDA_KERNEL_CALL((fused_forward_kernel_outdegree<float, 32, 5>), nblks,
                     nthrs, smem_size, h, f, nodes_subgraph,
                     smem_nodes_subgraph, store_node, store_flag, indptr,
                     indices, val, Q, K, V, out_feat);
    break;
  default:
    throw "not supported BLOCKSIZE!";
  }
}

void gf_forward_outdegree_multiple32(int num_subgraph, int h, int f,
                                     const int *nodes_subgraph,
                                     const int *smem_nodes_subgraph,
                                     const int *store_node,
                                     const int *store_flag, const int *indptr,
                                     const int *indices, const float *val,
                                     const float *Q, const float *K,
                                     const float *V, float *out_feat) {
  const int ntx = f;        // on feature dimension
  const int nty = 1024 / f; // on out dimension
  const int nbx = num_subgraph;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const int smem_size = 1024 * 64 - nty * 32 * 4;
  // printf("launch dim %d %d %d %d \n", ntx, nty, nbx, nby);
  switch (nty) {
  case 8:
    cudaFuncSetAttribute(fused_forward_kernel_outdegree_mul32<float, 8, 3>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    CUDA_KERNEL_CALL((fused_forward_kernel_outdegree_mul32<float, 8, 3>), nblks,
                     nthrs, smem_size, h, f, nodes_subgraph,
                     smem_nodes_subgraph, store_node, store_flag, indptr,
                     indices, val, Q, K, V, out_feat);
    break;
  case 16:
    cudaFuncSetAttribute(fused_forward_kernel_outdegree_mul32<float, 16, 4>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    CUDA_KERNEL_CALL((fused_forward_kernel_outdegree_mul32<float, 16, 4>),
                     nblks, nthrs, smem_size, h, f, nodes_subgraph,
                     smem_nodes_subgraph, store_node, store_flag, indptr,
                     indices, val, Q, K, V, out_feat);
    break;
  case 32:
    cudaFuncSetAttribute(fused_forward_kernel_outdegree_mul32<float, 32, 5>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);
    CUDA_KERNEL_CALL((fused_forward_kernel_outdegree_mul32<float, 32, 5>),
                     nblks, nthrs, smem_size, h, f, nodes_subgraph,
                     smem_nodes_subgraph, store_node, store_flag, indptr,
                     indices, val, Q, K, V, out_feat);
    break;
  default:
    throw "not supported BLOCKSIZE!";
  }
}

std::vector<torch::Tensor> gf_outdegree_forward_cuda(
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

  // check whether f is multiples of 32
  if (isMul32(f)) {
    gf_forward_outdegree_multiple32(
        num_subgraph, h, f, nodes_subgraph.data_ptr<int>(),
        smem_nodes_subgraph.data_ptr<int>(), store_node.data_ptr<int>(),
        store_flag.data_ptr<int>(), indptr.data_ptr<int>(),
        indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
        K.data_ptr<float>(), V.data_ptr<float>(), out_feat.data_ptr<float>());
  } else {
    gf_forward_outdegree(
        num_subgraph, h, f, nodes_subgraph.data_ptr<int>(),
        smem_nodes_subgraph.data_ptr<int>(), store_node.data_ptr<int>(),
        store_flag.data_ptr<int>(), indptr.data_ptr<int>(),
        indices.data_ptr<int>(), val.data_ptr<float>(), Q.data_ptr<float>(),
        K.data_ptr<float>(), V.data_ptr<float>(), out_feat.data_ptr<float>());
  }

  return {out_feat};
}
#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

__global__ void
fused_forward_ell_kernel(const int m, const int nnz, const int h, const int f,
                         const int *indptr, const int *indices,
                         const int *row_index, const int *rows_per_tb,
                         const float *val, const float *Q, const float *K,
                         const float *V, float *out_feat) {
  const int bid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim
  const int lrow = rows_per_tb[bid];              // row rid elements
  const int hrow = rows_per_tb[bid + 1];
  const int num_rows = hrow - lrow;
  extern __shared__ float smem[];
  float *curr_node_feature = smem;
  float *neigh_nodes_weight = (float *)&curr_node_feature[f];

  const int threads_x = blockDim.x; // 32
  const int threads_y = blockDim.y; // f/32
  const int blockSize = threads_x * threads_y;
  float weightMax = -1e38;

  for (int row = 0; row < num_rows; row++) {
    int rid = row_index[lrow + row];
    const int lb = indptr[rid]; // row rid elements
    const int hb = indptr[rid + 1];

    const int num_neighbor = hb - lb;

    // init the shared memory
    if (fid < f) {
      curr_node_feature[fid] = Q[rid * h * f + hid * f + fid];
    }

    // compute the attention weight
    for (int j = 0; j < num_neighbor; j++) {
      float weight = 0;
      float weight_partial = 0;
      if (fid < f) {
        int cid = indices[lb + j];
        weight_partial =
            curr_node_feature[fid] * K[cid * h * f + hid * f + fid];
      }
      __syncthreads();
      static __shared__ float warpLevelSums[WARP_SIZE];
      const int laneId = fid % WARP_SIZE;
      const int warpId = fid / WARP_SIZE;
      weight_partial = warpReduceSum(weight_partial, blockSize);
      if (laneId == 0)
        warpLevelSums[warpId] = weight_partial;
      __syncthreads();
      weight_partial =
          (fid < blockSize / WARP_SIZE) ? warpLevelSums[laneId] : 0;
      if (warpId == 0)
        weight_partial = warpReduceSum(weight_partial, blockSize / WARP_SIZE);
      if (fid == 0) {
        neigh_nodes_weight[j] = weight_partial;
      }
      __syncthreads();
      weight = neigh_nodes_weight[j];
      weightMax = MAX(weight, weightMax);
    }
    __syncthreads();

    // compute the sum of exp
    int loop = (num_neighbor + 31) / 32;
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
    __syncthreads();

    // compute the output
    float acc = 0;
    for (int j = 0; j < num_neighbor; j++) {
      float attn_val;
      int cid = indices[lb + j];
      if (fid < f) {
        float weight = neigh_nodes_weight[j];
        attn_val = exp(weight - weightMax) / expAll;
        acc += attn_val * V[cid * h * f + hid * f + fid];
      }
      __syncthreads();
    }
    if (fid < f)
      out_feat[rid * h * f + hid * f + fid] = acc;
    __syncthreads();
  }
}

void gf_ell_forward(int m, int nnz, int h, int f, int num_tb, const int *indptr,
                    const int *indices, const int *row_index,
                    const int *rows_per_tb, const float *val, const float *Q,
                    const float *K, const float *V, float *out_feat) {
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);
  const int ntx = WARP_SIZE;
  const int nty = (f + WARP_SIZE - 1) / WARP_SIZE;

  const dim3 nblks(num_tb, h);
  const dim3 nthrs(ntx, nty);
  CUDA_KERNEL_CALL((fused_forward_ell_kernel), nblks, nthrs,
                   (f + 512) * sizeof(float), m, nnz, h, f, indptr, indices,
                   row_index, rows_per_tb, val, Q, K, V, out_feat);
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // float elapsedTime;
  // cudaEventElapsedTime(&elapsedTime, start, stop);
  // printf("Time of fused kernel: %f \n", elapsedTime);
}

std::vector<torch::Tensor>
gf_ell_forward_cuda(torch::Tensor indptr, torch::Tensor indices,
                    torch::Tensor row_index, torch::Tensor rows_per_tb,
                    torch::Tensor val, torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto m = indptr.size(0) - 1;           // num of nodes
  const auto nnz = indices.size(0);            // num of edges
  const auto h = Q.size(1);                    // num of heads
  const auto f = Q.size(2);                    // num of feats
  const auto num_tb = rows_per_tb.size(0) - 1; // num of thread blocks
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);
  gf_ell_forward(m, nnz, h, f, num_tb, indptr.data_ptr<int>(),
                 indices.data_ptr<int>(), row_index.data_ptr<int>(),
                 rows_per_tb.data_ptr<int>(), val.data_ptr<float>(),
                 Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                 out_feat.data_ptr<float>());
  return {out_feat};
}
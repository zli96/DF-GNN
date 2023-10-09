#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

__global__ void fused_forward_kernel_mul32(const int m, const int nnz,
                                           const int h, const int f,
                                           const int *indptr,
                                           const int *indices, const float *val,
                                           const float *Q, const float *K,
                                           const float *V, float *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  extern __shared__ float smem[];
  float *neigh_nodes_weight = smem;
  float weightMax = -1e38;
  static __shared__ float warpLevelSums[WARP_SIZE];
  const int hf = h * f;
  const int hfid = hid * f + fid;
  const int laneId = threadIdx.x;
  const int warpId = threadIdx.y;
  float Q_i = Q[rid * hf + hfid];

  for (int j = 0; j < num_neighbor; j++) {
    float weight = 0;
    float weight_partial = 0;

    int cid = indices[lb + j];
    weight_partial = Q_i * K[cid * hf + hfid];
    __syncwarp();

    weight_partial = warpReduceSum(weight_partial, f);
    if (laneId == 0)
      warpLevelSums[warpId] = weight_partial;
    __syncthreads();
    weight_partial = (fid < f / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if (warpId == 0)
      weight_partial = warpReduceSum(weight_partial, f / WARP_SIZE);
    if (fid == 0) {
      neigh_nodes_weight[j] = weight_partial * val[lb + j];
    }
    __syncthreads();
    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
  }

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

  // compute the output
  float acc = 0;
  for (int j = 0; j < num_neighbor; j++) {
    float attn_val;
    int cid = indices[lb + j];
    float weight = neigh_nodes_weight[j];
    attn_val = exp(weight - weightMax);
    acc += attn_val * V[cid * hf + hfid];
  }

  // handle the node with no neighbor
  out_feat[rid * hf + hfid] = (expAll != 0) ? acc / expAll : 0;
}

__global__ void fused_forward_kernel_tiling_mul32(
    const int m, const int nnz, const int h, const int f, const int *indptr,
    const int *indices, const float *val, const float *Q, const float *K,
    const float *V, float *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  extern __shared__ float smem[];
  float *neigh_nodes_weight = smem;
  static __shared__ float warpLevelSums[WARP_SIZE];
  const int hf = h * f;
  const int hfid = hid * f + fid;
  const int laneId = threadIdx.x;
  const int warpId = threadIdx.y;
  float Q_i = Q[rid * hf + hfid];

  float acc = 0, partial_sum = 0;
  float weightMax_old = -1e38, weightMax = -1e38;
  float expweight, expweightMax;

  for (int j = 0; j < num_neighbor; j++) {
    float weight = 0;
    float weight_partial = 0;

    int cid = indices[lb + j];
    weight_partial = Q_i * K[cid * hf + hfid];
    __syncwarp();

    weight_partial = warpReduceSum(weight_partial, f);
    if (laneId == 0)
      warpLevelSums[warpId] = weight_partial;
    __syncthreads();
    weight_partial = (fid < f / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if (warpId == 0)
      weight_partial = warpReduceSum(weight_partial, f / WARP_SIZE);
    if (fid == 0) {
      neigh_nodes_weight[j] = weight_partial * val[lb + j];
    }
    __syncthreads();
    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
    expweight = exp(weight - weightMax);
    expweightMax =
        (weightMax_old == weightMax) ? 1 : exp(weightMax_old - weightMax);
    acc = acc * expweightMax + expweight * V[cid * hf + hfid];
    partial_sum = partial_sum * expweightMax + expweight;
    weightMax_old = weightMax;
  }

  // handle the node with no neighbor
  out_feat[rid * hf + hfid] = (partial_sum != 0) ? acc / partial_sum : 0;
}

__global__ void fused_forward_kernel(const int m, const int nnz, const int h,
                                     const int f, const int *indptr,
                                     const int *indices, const float *val,
                                     const float *Q, const float *K,
                                     const float *V, float *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int laneId = fid % WARP_SIZE;
  const int warpId = fid / WARP_SIZE;

  const int f_mul_32 = roundup(f, 32);
  const int num_neighbor = hb - lb;

  // Allocate smem
  static __shared__ float warpLevelSums[WARP_SIZE];
  extern __shared__ float smem[];
  float *neigh_nodes_weight = smem;
  float weightMax = -1e38;

  // init the shared memory
  float Q_i = 0;
  if (fid < f) {
    Q_i = Q[rid * h * f + hid * f + fid];
  }

  // compute the attention weight
  for (int j = 0; j < num_neighbor; j++) {
    float weight = 0;
    float weight_partial = 0;
    if (fid < f) {
      int cid = indices[lb + j];
      weight_partial = Q_i * K[cid * h * f + hid * f + fid];
    }
    __syncwarp();

    weight_partial = warpReduceSum(weight_partial, f_mul_32);
    if (laneId == 0)
      warpLevelSums[warpId] = weight_partial;
    namedBarrierSync(0, f_mul_32);

    weight_partial = (fid < f_mul_32 / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if (warpId == 0)
      weight_partial = warpReduceSum(weight_partial, f_mul_32 / WARP_SIZE);
    if (fid == 0) {
      neigh_nodes_weight[j] = weight_partial;
    }

    namedBarrierSync(0, f_mul_32);
    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
  }

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

  // compute the output
  float acc = 0;
  for (int j = 0; j < num_neighbor; j++) {
    float attn_val;
    int cid = indices[lb + j];
    float weight = neigh_nodes_weight[j];
    attn_val = exp(weight - weightMax);
    if (fid < f) {
      acc += attn_val * V[cid * h * f + hid * f + fid];
    }
  }
  if (fid < f)
    // handle the node with no neighbor
    out_feat[rid * h * f + hid * f + fid] = (expAll != 0) ? acc / expAll : 0;
}

void gf_forward(int m, int nnz, int h, int f, int smem_consume,
                const int *indptr, const int *indices, const float *val,
                const float *Q, const float *K, const float *V,
                float *out_feat) {
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);
  const int ntx = WARP_SIZE;
  const int nty = (f + WARP_SIZE - 1) / WARP_SIZE;

  const dim3 nblks(m, h);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL((fused_forward_kernel), nblks, nthrs,
                   (smem_consume) * sizeof(float), m, nnz, h, f, indptr,
                   indices, val, Q, K, V, out_feat);
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // float elapsedTime;
  // cudaEventElapsedTime(&elapsedTime, start, stop);
  // printf("Time of fused kernel: %f \n", elapsedTime);
}

void gf_forward_multiple32(int m, int nnz, int h, int f, int smem_consume,
                           const int *indptr, const int *indices,
                           const float *val, const float *Q, const float *K,
                           const float *V, float *out_feat) {
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);
  const dim3 nblks(m, h, 1);
  const dim3 nthrs(32, f / 32, 1);
  CUDA_KERNEL_CALL((fused_forward_kernel_mul32), nblks, nthrs,
                   (smem_consume) * sizeof(float), m, nnz, h, f, indptr,
                   indices, val, Q, K, V, out_feat);
}

std::vector<torch::Tensor> gf_forward_cuda(torch::Tensor indptr,
                                           torch::Tensor indices,
                                           torch::Tensor val, int smem_consume,
                                           torch::Tensor Q, torch::Tensor K,
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

  // check whether f is multiples of 32
  if (isMul32(f)) {
    gf_forward_multiple32(m, nnz, h, f, smem_consume, indptr.data_ptr<int>(),
                          indices.data_ptr<int>(), val.data_ptr<float>(),
                          Q.data_ptr<float>(), K.data_ptr<float>(),
                          V.data_ptr<float>(), out_feat.data_ptr<float>());
  } else {
    gf_forward(m, nnz, h, f, smem_consume, indptr.data_ptr<int>(),
               indices.data_ptr<int>(), val.data_ptr<float>(),
               Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
               out_feat.data_ptr<float>());
  }
  return {out_feat};
}

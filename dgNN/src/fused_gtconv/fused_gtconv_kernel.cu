#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType>
__global__ void
fused_forward_kernel_mul32(const int m, const int nnz, const int h, const int f,
                           const int *indptr, const int *indices,
                           const DType *val, const DType *Q, const DType *K,
                           const DType *V, DType *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem;
  DType weightMax = -1e38;
  static __shared__ DType warpLevelSums[WARP_SIZE];
  const int hf = h * f;
  const int hfid = hid * f + fid;
  const int laneId = threadIdx.x;
  const int warpId = threadIdx.y;
  const DType Q_i = Q[rid * hf + hfid];
  const DType *valoff = val + lb;

  for (int j = 0; j < num_neighbor; j++) {
    DType weight = 0;
    DType weight_partial = 0;

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
      neigh_nodes_weight[j] = weight_partial * valoff[j];
    }
    __syncthreads();
    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
  }

  // compute the sum of exp
  int loop = (num_neighbor + 31) / 32;
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
  DType attn_val;
  for (int j = 0; j < num_neighbor; j++) {
    int cid = indices[lb + j];
    DType weight = neigh_nodes_weight[j];
    attn_val = exp(weight - weightMax);
    acc += attn_val * V[cid * hf + hfid];
  }

  // handle the node with no neighbor
  out_feat[rid * hf + hfid] = (expAll != 0) ? acc / expAll : 0;
}

template <typename DType>
__global__ void fused_forward_kernel_tiling_mul32(
    const int m, const int nnz, const int h, const int f, const int *indptr,
    const int *indices, const DType *val, const DType *Q, const DType *K,
    const DType *V, DType *out_feat) {
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int num_neighbor = hb - lb;
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem;
  static __shared__ DType warpLevelSums[WARP_SIZE];
  const int hf = h * f;
  const int hfid = hid * f + fid;
  const int laneId = threadIdx.x;
  const int warpId = threadIdx.y;
  DType Q_i = Q[rid * hf + hfid];

  DType acc = 0, partial_sum = 0;
  DType weightMax_old = -1e38, weightMax = -1e38;
  DType expweight, expweightMax;

  for (int j = 0; j < num_neighbor; j++) {
    DType weight = 0;
    DType weight_partial = 0;

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

template <typename DType>
__global__ void fused_forward_kernel(const int m, const int nnz, const int h,
                                     const int f, const int *indptr,
                                     const int *indices, const DType *val,
                                     const DType *Q, const DType *K,
                                     const DType *V, DType *out_feat) {
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
  static __shared__ DType warpLevelSums[WARP_SIZE];
  extern __shared__ DType smem[];
  DType *neigh_nodes_weight = smem;
  DType weightMax = -1e38;
  const DType *valoff = val + lb;
  // init the shared memory
  DType Q_i = 0;
  if (fid < f) {
    Q_i = Q[rid * h * f + hid * f + fid];
  }

  // compute the attention weight
  for (int j = 0; j < num_neighbor; j++) {
    DType weight = 0;
    DType weight_partial = 0;
    if (fid < f) {
      int cid = indices[lb + j];
      weight_partial = Q_i * K[cid * h * f + hid * f + fid];
    }
    __syncwarp();

    weight_partial = warpReduceSum(weight_partial, f_mul_32);
    if (laneId == 0)
      warpLevelSums[warpId] = weight_partial;
    __syncthreads();

    weight_partial = (fid < f_mul_32 / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if (warpId == 0)
      weight_partial = warpReduceSum(weight_partial, f_mul_32 / WARP_SIZE);
    if (fid == 0) {
      neigh_nodes_weight[j] = weight_partial * valoff[j];
    }
    __syncthreads();

    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
  }

  // compute the sum of exp
  int loop = (num_neighbor + 31) / 32;
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
  DType attn_val;
  for (int j = 0; j < num_neighbor; j++) {
    int cid = indices[lb + j];
    DType weight = neigh_nodes_weight[j];
    attn_val = exp(weight - weightMax);
    if (fid < f) {
      acc += attn_val * V[cid * h * f + hid * f + fid];
    }
  }
  if (fid < f)
    // handle the node with no neighbor
    out_feat[rid * h * f + hid * f + fid] = (expAll != 0) ? acc / expAll : 0;
}

void gt_forward(int m, int nnz, int h, int f, int smem_consume,
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

  CUDA_KERNEL_CALL((fused_forward_kernel<float>), nblks, nthrs,
                   (smem_consume) * sizeof(float), m, nnz, h, f, indptr,
                   indices, val, Q, K, V, out_feat);
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // float elapsedTime;
  // cudaEventElapsedTime(&elapsedTime, start, stop);
  // printf("Time of fused kernel: %f \n", elapsedTime);
}

void gt_forward_multiple32(int m, int nnz, int h, int f, int smem_consume,
                           const int *indptr, const int *indices,
                           const float *val, const float *Q, const float *K,
                           const float *V, float *out_feat) {
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);
  const dim3 nblks(m, h, 1);
  const dim3 nthrs(32, f / 32, 1);
  CUDA_KERNEL_CALL((fused_forward_kernel_mul32<float>), nblks, nthrs,
                   (smem_consume) * sizeof(float), m, nnz, h, f, indptr,
                   indices, val, Q, K, V, out_feat);
}

std::vector<torch::Tensor> gt_forward_cuda(torch::Tensor indptr,
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
    gt_forward_multiple32(m, nnz, h, f, smem_consume, indptr.data_ptr<int>(),
                          indices.data_ptr<int>(), val.data_ptr<float>(),
                          Q.data_ptr<float>(), K.data_ptr<float>(),
                          V.data_ptr<float>(), out_feat.data_ptr<float>());
  } else {
    gt_forward(m, nnz, h, f, smem_consume, indptr.data_ptr<int>(),
               indices.data_ptr<int>(), val.data_ptr<float>(),
               Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
               out_feat.data_ptr<float>());
  }
  return {out_feat};
}
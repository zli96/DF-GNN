#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>
#include <unistd.h>
#include <stdio.h>

using namespace std;

extern "C" bool isMul32(int x)
{
  return (x >= 0 && x % 32 == 0);
}

#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

#define CUSPARSE_CALL(func)                                         \
  {                                                                 \
    cusparseStatus_t e = (func);                                    \
    CHECK(e == CUSPARSE_STATUS_SUCCESS) << "CUSPARSE ERROR: " << e; \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, ...)          \
  {                                                                 \
    {                                                               \
      (kernel)<<<(nblks), (nthrs), (shmem)>>>(__VA_ARGS__);         \
      cudaError_t e = cudaGetLastError();                           \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)      \
          << "CUDA kernel launch error: " << cudaGetErrorString(e); \
    }                                                               \
  }

__device__ __forceinline__ float warpReduceSum(float sum, int blockSize)
{
  if (blockSize >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);
  if (blockSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);
  if (blockSize >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);
  if (blockSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);
  if (blockSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

template <typename DType>
__global__ void sddmmCooKernel(const int lhs_len, const int rhs_len, const int out_len,
                               const int nnz, const int reduce_size,
                               const int *row, const int *col, const DType *data,
                               const DType *lhs, const DType *rhs, DType *out)
{
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  if (ty < nnz)
  {
    const int src = __ldg(row + ty);
    const int dst = __ldg(col + ty);
    const int eid = ty;
    const DType *lhsoff = lhs + src * lhs_len;
    const DType *rhsoff = rhs + dst * rhs_len;
    DType *outoff = out + eid * out_len;
    int tx = threadIdx.x; // tx < 32
    for (int i = blockIdx.y; i < out_len; i += gridDim.y)
    { // over output feature dimension
      DType val = 0;
      for (int j = tx; j < reduce_size; j += 64)
      {
        val += lhsoff[i * reduce_size + j] *
               rhsoff[i * reduce_size + j];
        if (j + 32 < reduce_size)
          val += lhsoff[i * reduce_size + j + 32] *
                 rhsoff[i * reduce_size + j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(full_mask, val, offset);
      if (tx == 0)
      {
        outoff[i] = val;
      }
    }
  }
}

__global__ void sddmmCsrKernel(const int m, const int nnz, const int h, const int f,
                               const int *indptr, const int *indices, const float *val,
                               const float *Q, const float *K, float *attn_edge)
{
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

  for (int j = 0; j < num_neighbor; j++)
  {
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
    if (fid == 0)
    {
      attn_edge[lb + j] = weight_partial * val[lb + j];
    }
  }
  __syncthreads();
}

__global__ void softMax_SPMM(const int m, const int nnz, const int h, const int f,
                             const int *indptr, const int *indices, const float *val,
                             const float *V, const float *attn_edge,
                             float *out_feat)
{
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

  for (int j = 0; j < num_neighbor; j++)
  {
    float weight = 0;
    if (fid == 0)
    {
      neigh_nodes_weight[j] = attn_edge[lb + j];
    }
    __syncthreads();
    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
  }
  // compute the sum of exp
  int loop = (num_neighbor + 31) / 32;
  float expAll = 0;
  for (int j = 0; j < loop; j++)
  {
    int pid = threadIdx.x + (j << 5); // node need to process in loop j
    float exptmp = 0;
    if (pid < num_neighbor)
    {
      float weight = neigh_nodes_weight[pid];
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
    }
    __syncwarp();
    expAll += exptmp;
  }
  __syncthreads();

  // compute the output
  float acc = 0;
  for (int j = 0; j < num_neighbor; j++)
  {
    float attn_val;
    int cid = indices[lb + j];
    float weight = neigh_nodes_weight[j];
    attn_val = exp(weight - weightMax) / expAll;
    acc += attn_val * V[cid * hf + hfid];
    __syncthreads();
  }

  out_feat[rid * hf + hfid] = acc;
}

template <typename DType, int blockSize, int LOG_BLOCK_SIZE>
__global__ void fused_forward_kernel_subgraph(const int h, const int f,
                                              const int *node_num_ptr, const int *indptr, const int *indices, const DType *val,
                                              const DType *Q, const DType *K, const DType *V,
                                              DType *out_feat)
{
  // grid: 4096*h  block: 32 * 8 each tb processes one graph
  // blockSize = blockDim.y
  const int gid = blockIdx.x;   // index of subgraph
  const int hid = blockIdx.y;   // index of head
  const int tidx = threadIdx.x; // index of WARP
  const int tidy = threadIdx.y;
  const int node_lb = node_num_ptr[gid];
  const int node_hb = node_num_ptr[gid + 1]; // Offset of nodes in the subgraph on the full graph
  const int num_nodes = node_hb - node_lb;   // num of nodes in this subgraph

  const int hf = h * f;
  extern __shared__ DType smem[];
  DType *K_SMEM = smem;
  DType *V_SMEM = (DType *)&K_SMEM[num_nodes * hf];
  DType *neigh_nodes_weight = (DType *)&V_SMEM[num_nodes * hf];

  int loops_node = (num_nodes + blockSize - 1) / blockSize;
  int loops_feat = (f + WARP_SIZE - 1) / WARP_SIZE;
  // Put the K and V into smem
  for (int j = 0; j < loops_node; j++)
  {
    int curr_node = j * blockSize + tidy;
    if (curr_node + node_lb < node_hb)
    {

      for (int i = 0; i < loops_feat; i++)
      {
        int curr_feat = tidx + (i << 5);
        if (curr_feat < f)
        {
          K_SMEM[curr_node * hf + hid * f + curr_feat] = K[(node_lb + curr_node) * hf + hid * f + curr_feat];
          V_SMEM[curr_node * hf + hid * f + curr_feat] = V[(node_lb + curr_node) * hf + hid * f + curr_feat];
        }
      }
    }
  }

  __syncthreads();

  for (int j = 0; j < loops_node; j++)
  {
    int curr_node = tidy + j * blockSize;
    if (curr_node + node_lb < node_hb)
    {
      DType weightMax = -1e38;
      const int lb = indptr[node_lb + curr_node]; // row rid elements
      const int hb = indptr[node_lb + curr_node + 1];
      const int num_neighbor = hb - lb;
      for (int k = 0; k < num_neighbor; k++)
      {
        int cid = indices[lb + k] - node_lb;
        DType weight_partial = 0;
        for (int i = 0; i < loops_feat; i++)
        {
          int curr_feat = tidx + (i << 5);
          if (curr_feat < f)
          {
            DType Q_i = Q[(node_lb + curr_node) * hf + hid * f + curr_feat];
            weight_partial += Q_i * K_SMEM[cid * hf + hid * f + curr_feat];
          }
        }
        __syncthreads();
        // weight_partial = warpReduceSum(weight_partial, WARP_SIZE);
        for (int stride = 16; stride > 0; stride >>= 1)
        {
          weight_partial += __shfl_xor_sync(0xffffffff, weight_partial, stride, 32);
        }
        __syncwarp();
        if (tidx == 0)
        {
          neigh_nodes_weight[tidy + (k << LOG_BLOCK_SIZE)] = weight_partial * val[lb + k];
        }
        __syncthreads();
        weightMax = MAX(weight_partial, weightMax);
      }
      __syncthreads();

      int loop_WARP_neigh = (num_neighbor + WARP_SIZE - 1) / WARP_SIZE;
      DType expAll = 0;
      for (int k = 0; k < loop_WARP_neigh; k++)
      {
        DType exptmp = 0;
        int pid = tidx + (k << 5);
        if (pid < num_neighbor)
        {
          DType weight = neigh_nodes_weight[tidy + (pid << LOG_BLOCK_SIZE)];
          exptmp = exp(weight - weightMax);
        }
        __syncwarp();
        for (int stride = 16; stride > 0; stride >>= 1)
        {
          exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
        }
        __syncwarp();
        expAll += exptmp;
      }
      __syncthreads();

      // compute the output
      for (int i = 0; i < loops_feat; i++)
      {
        DType acc = 0;
        DType attn_val;
        int curr_feat = tidx + (i << 5);
        if (curr_feat < f)
        {
          for (int k = 0; k < num_neighbor; k++)
          {
            int cid = indices[lb + k] - node_lb;
            DType weight = neigh_nodes_weight[tidy + (k << LOG_BLOCK_SIZE)];
            attn_val = exp(weight - weightMax) / expAll;
            acc += attn_val * V_SMEM[cid * hf + hid * f + curr_feat];
          }
        }
        __syncthreads();
        out_feat[(node_lb + curr_node) * hf + hid * f + curr_feat] = acc;
      }
    }
    __syncthreads();
  }
}

__global__ void fused_forward_kernel_mul32(const int m, const int nnz, const int h, const int f,
                                           const int *indptr, const int *indices, const float *val,
                                           const float *Q, const float *K, const float *V,
                                           float *out_feat)
{
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
  const int laneId = fid % WARP_SIZE;
  const int warpId = fid / WARP_SIZE;
  float Q_i = Q[rid * hf + hfid];

  for (int j = 0; j < num_neighbor; j++)
  {
    float weight = 0;
    float weight_partial = 0;

    int cid = indices[lb + j];
    weight_partial = Q_i * K[cid * hf + hfid];

    __syncthreads();
    weight_partial = warpReduceSum(weight_partial, f);
    if (laneId == 0)
      warpLevelSums[warpId] = weight_partial;
    __syncthreads();
    weight_partial = (fid < f / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if (warpId == 0)
      weight_partial = warpReduceSum(weight_partial, f / WARP_SIZE);
    if (fid == 0)
    {
      neigh_nodes_weight[j] = weight_partial * val[lb + j];
    }
    __syncthreads();
    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
  }
  __syncthreads();

  // compute the sum of exp
  int loop = (num_neighbor + 31) / 32;
  float expAll = 0;
  for (int j = 0; j < loop; j++)
  {
    int pid = threadIdx.x + (j << 5); // node need to process in loop j
    float exptmp = 0;
    if (pid < num_neighbor)
    {
      float weight = neigh_nodes_weight[pid];
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
    }
    __syncwarp();
    expAll += exptmp;
  }
  __syncthreads();

  // compute the output
  float acc = 0;
  for (int j = 0; j < num_neighbor; j++)
  {
    float attn_val;
    int cid = indices[lb + j];
    float weight = neigh_nodes_weight[j];
    attn_val = exp(weight - weightMax) / expAll;
    acc += attn_val * V[cid * hf + hfid];
    __syncthreads();
  }

  out_feat[rid * hf + hfid] = acc;
}

__global__ void fused_forward_kernel(const int m, const int nnz, const int h, const int f,
                                     const int *indptr, const int *indices, const float *val,
                                     const float *Q, const float *K, const float *V,
                                     float *out_feat)
{
  const int rid = blockIdx.x;                     // loop over row of adj matrix
  const int hid = blockIdx.y;                     // loop over heads
  const int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  const int lb = indptr[rid]; // row rid elements
  const int hb = indptr[rid + 1];

  const int threads_x = blockDim.x; // 32
  const int threads_y = blockDim.y; // f/32
  const int blockSize = threads_x * threads_y;
  const int num_neighbor = hb - lb;
  extern __shared__ float smem[];
  float *curr_node_feature = smem;
  float *neigh_nodes_weight = (float *)&curr_node_feature[f];
  float weightMax = -1e38;
  // init the shared memory
  float Q_i = 0;
  if (fid < f)
  {
    Q_i = Q[rid * h * f + hid * f + fid];
  }

  // compute the attention weight
  for (int j = 0; j < num_neighbor; j++)
  {
    float weight = 0;
    float weight_partial = 0;
    if (fid < f)
    {
      int cid = indices[lb + j];
      weight_partial = Q_i * K[cid * h * f + hid * f + fid];
    }
    __syncthreads();
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = fid % WARP_SIZE;
    const int warpId = fid / WARP_SIZE;
    weight_partial = warpReduceSum(weight_partial, blockSize);
    if (laneId == 0)
      warpLevelSums[warpId] = weight_partial;
    __syncthreads();
    weight_partial = (fid < blockSize / WARP_SIZE) ? warpLevelSums[laneId] : 0;
    if (warpId == 0)
      weight_partial = warpReduceSum(weight_partial, blockSize / WARP_SIZE);
    if (fid == 0)
    {
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
  for (int j = 0; j < loop; j++)
  {
    int pid = threadIdx.x + (j << 5); // node need to process in loop j
    float exptmp = 0;
    if (pid < num_neighbor)
    {
      float weight = neigh_nodes_weight[pid];
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
    }
    __syncwarp();
    expAll += exptmp;
  }
  __syncthreads();

  // compute the output
  float acc = 0;
  for (int j = 0; j < num_neighbor; j++)
  {
    float attn_val;
    int cid = indices[lb + j];
    if (fid < f)
    {
      float weight = neigh_nodes_weight[j];
      attn_val = exp(weight - weightMax) / expAll;
      acc += attn_val * V[cid * h * f + hid * f + fid];
    }
    __syncthreads();
  }
  if (fid < f)
    out_feat[rid * h * f + hid * f + fid] = acc;
}

__global__ void fused_forward_ell_kernel(const int m, const int nnz, const int h, const int f,
                                         const int *indptr, const int *indices,
                                         const int *row_index, const int *rows_per_tb, const float *val,
                                         const float *Q, const float *K, const float *V,
                                         float *out_feat)
{
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

  for (int row = 0; row < num_rows; row++)
  {
    int rid = row_index[lrow + row];
    const int lb = indptr[rid]; // row rid elements
    const int hb = indptr[rid + 1];

    const int num_neighbor = hb - lb;

    // init the shared memory
    if (fid < f)
    {
      curr_node_feature[fid] = Q[rid * h * f + hid * f + fid];
    }

    // compute the attention weight
    for (int j = 0; j < num_neighbor; j++)
    {
      float weight = 0;
      float weight_partial = 0;
      if (fid < f)
      {
        int cid = indices[lb + j];
        weight_partial = curr_node_feature[fid] * K[cid * h * f + hid * f + fid];
      }
      __syncthreads();
      static __shared__ float warpLevelSums[WARP_SIZE];
      const int laneId = fid % WARP_SIZE;
      const int warpId = fid / WARP_SIZE;
      weight_partial = warpReduceSum(weight_partial, blockSize);
      if (laneId == 0)
        warpLevelSums[warpId] = weight_partial;
      __syncthreads();
      weight_partial = (fid < blockSize / WARP_SIZE) ? warpLevelSums[laneId] : 0;
      if (warpId == 0)
        weight_partial = warpReduceSum(weight_partial, blockSize / WARP_SIZE);
      if (fid == 0)
      {
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
    for (int j = 0; j < loop; j++)
    {
      int pid = threadIdx.x + (j << 5); // node need to process in loop j
      float exptmp = 0;
      if (pid < num_neighbor)
      {
        float weight = neigh_nodes_weight[pid];
        exptmp = exp(weight - weightMax);
      }
      __syncwarp();
      for (int stride = 16; stride > 0; stride >>= 1)
      {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }
    __syncthreads();

    // compute the output
    float acc = 0;
    for (int j = 0; j < num_neighbor; j++)
    {
      float attn_val;
      int cid = indices[lb + j];
      if (fid < f)
      {
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

void gf_forward(int m, int nnz, int h, int f,
                const int *indptr, const int *indices, const float *val,
                const float *Q, const float *K, const float *V,
                float *out_feat)
{
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);

  const dim3 nblks(m, h, 1);
  const dim3 nthrs(32, (f + 31) / 32, 1);
  CUDA_KERNEL_CALL(
      (fused_forward_kernel),
      nblks, nthrs, (f + 512) * sizeof(float), m, nnz, h, f, indptr, indices, val,
      Q, K, V, out_feat);
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // float elapsedTime;
  // cudaEventElapsedTime(&elapsedTime, start, stop);
  // printf("Time of fused kernel: %f \n", elapsedTime);
}

void gf_forward_subgraph(int num_subgraph, int h, int f, const int *nodes_subgraph,
                         const int *indptr, const int *indices, const float *val,
                         const float *Q, const float *K, const float *V,
                         float *out_feat)
{
  const int BLOCK_SIZE = atoi(getenv("BLOCK_SIZE"));

  const int ntx = 32;         // on feature dimension
  const int nty = BLOCK_SIZE; // on out dimension
  const int nbx = num_subgraph;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  switch (BLOCK_SIZE)
  {
  case 8:
    cudaFuncSetAttribute(fused_forward_kernel_subgraph<float, 8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    CUDA_KERNEL_CALL(
        (fused_forward_kernel_subgraph<float, 8, 3>),
        nblks, nthrs, 1024 * 64, h, f, nodes_subgraph, indptr, indices, val,
        Q, K, V, out_feat);
    break;
  case 32:
    cudaFuncSetAttribute(fused_forward_kernel_subgraph<float, 32, 5>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    CUDA_KERNEL_CALL(
        (fused_forward_kernel_subgraph<float, 32, 5>),
        nblks, nthrs, 1024 * 64, h, f, nodes_subgraph, indptr, indices, val,
        Q, K, V, out_feat);
    break;
  default:
    throw "not supported BLOCKSIZE!";
  }
}

void gf_forward_nofuse(int m, int nnz, int h, int f,
                       const int *indptr, const int *indices, const int *rows, const float *val,
                       const float *Q, const float *K, const float *V,
                       float *attn_edge, float *out_feat)
{
  const int ntx = 32; // on feature dimension
  const int nty = 8;  // on out dimension
  const int nbx = (nnz + nty - 1) / nty;
  const int nby = FindNumBlocks<'y'>(h);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL(
      (sddmmCooKernel<float>),
      nblks, nthrs, 0, f * h, f * h, h, nnz, f, rows, indices, val,
      Q, K, attn_edge);

  const dim3 nblks2(m, h, 1);
  const dim3 nthrs2(32, (f + 31) / 32, 1);
  // CUDA_KERNEL_CALL(
  //     (sddmmCsrKernel),
  //     nblks2, nthrs2, (f + 512) * sizeof(float), m, nnz, h, f, indptr, indices, val,
  //     Q, K, attn_edge);

  // const dim3 nblks2(m, h, 1);
  // const dim3 nthrs2(32, (f + 31) / 32, 1);
  CUDA_KERNEL_CALL(
      (softMax_SPMM),
      nblks2, nthrs2, (f + 512) * sizeof(float), m, nnz, h, f, indptr, indices, val,
      V, attn_edge, out_feat);
}

void gf_forward_multiple32(int m, int nnz, int h, int f,
                           const int *indptr, const int *indices, const float *val,
                           const float *Q, const float *K, const float *V,
                           float *out_feat)
{
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);
  const dim3 nblks(m, h, 1);
  const dim3 nthrs(32, f / 32, 1);
  CUDA_KERNEL_CALL(
      (fused_forward_kernel_mul32),
      nblks, nthrs, (512) * sizeof(float), m, nnz, h, f, indptr, indices, val,
      Q, K, V, out_feat);
}

void gf_ell_forward(int m, int nnz, int h, int f, int num_tb,
                    const int *indptr, const int *indices,
                    const int *row_index, const int *rows_per_tb, const float *val,
                    const float *Q, const float *K, const float *V,
                    float *out_feat)
{
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);

  const dim3 nblks(num_tb, h, 1);
  const dim3 nthrs(32, (f + 31) / 32, 1);
  CUDA_KERNEL_CALL(
      (fused_forward_ell_kernel),
      nblks, nthrs, (f + 512) * sizeof(float), m, nnz, h, f, indptr, indices, row_index, rows_per_tb, val,
      Q, K, V, out_feat);
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // float elapsedTime;
  // cudaEventElapsedTime(&elapsedTime, start, stop);
  // printf("Time of fused kernel: %f \n", elapsedTime);
}

std::vector<torch::Tensor>
gf_forward_cuda(torch::Tensor indptr,
                torch::Tensor indices,
                torch::Tensor val, torch::Tensor Q,
                torch::Tensor K, torch::Tensor V)
{
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
  if (isMul32(f))
  {
    gf_forward_multiple32(m, nnz, h, f,
                          indptr.data_ptr<int>(), indices.data_ptr<int>(), val.data_ptr<float>(),
                          Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                          out_feat.data_ptr<float>());
  }
  else
  {
    gf_forward(m, nnz, h, f,
               indptr.data_ptr<int>(), indices.data_ptr<int>(), val.data_ptr<float>(),
               Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
               out_feat.data_ptr<float>());
  }
  return {out_feat};
}

std::vector<torch::Tensor>
gf_subgraph_forward_cuda(torch::Tensor nodes_subgraph,
                         torch::Tensor indptr,
                         torch::Tensor indices,
                         torch::Tensor val, torch::Tensor Q,
                         torch::Tensor K, torch::Tensor V)
{
  // Q: torch.Size([6248, 10, 8])
  const auto num_subgraph = nodes_subgraph.size(0) - 1;
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);
  gf_forward_subgraph(num_subgraph, h, f, nodes_subgraph.data_ptr<int>(),
                      indptr.data_ptr<int>(), indices.data_ptr<int>(), val.data_ptr<float>(),
                      Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                      out_feat.data_ptr<float>());

  return {out_feat};
}

std::vector<torch::Tensor>
gf_hyper_forward_cuda(torch::Tensor indptr,
                      torch::Tensor indices, torch::Tensor rows,
                      torch::Tensor val, torch::Tensor Q,
                      torch::Tensor K, torch::Tensor V)
{
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
  gf_forward_nofuse(m, nnz, h, f,
                    indptr.data_ptr<int>(), indices.data_ptr<int>(), rows.data_ptr<int>(), val.data_ptr<float>(),
                    Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                    attn_edge.data_ptr<float>(), out_feat.data_ptr<float>());

  return {out_feat};
}

std::vector<torch::Tensor>
gf_ell_forward_cuda(torch::Tensor indptr,
                    torch::Tensor indices,
                    torch::Tensor row_index,
                    torch::Tensor rows_per_tb,
                    torch::Tensor val, torch::Tensor Q,
                    torch::Tensor K, torch::Tensor V)
{
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
  gf_ell_forward(m, nnz, h, f, num_tb,
                 indptr.data_ptr<int>(), indices.data_ptr<int>(),
                 row_index.data_ptr<int>(), rows_per_tb.data_ptr<int>(), val.data_ptr<float>(),
                 Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
                 out_feat.data_ptr<float>());
  return {out_feat};
}
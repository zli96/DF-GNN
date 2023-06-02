#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>

#include <unistd.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

#define CURAND_CALL(x)                                \
  do                                                  \
  {                                                   \
    if ((x) != CURAND_STATUS_SUCCESS)                 \
    {                                                 \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      return EXIT_FAILURE;                            \
    }                                                 \
  } while (0)

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, ...)          \
  {                                                                 \
    {                                                               \
      (kernel)<<<(nblks), (nthrs), (shmem)>>>(__VA_ARGS__);         \
      cudaError_t e = cudaGetLastError();                           \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)      \
          << "CUDA kernel launch error: " << cudaGetErrorString(e); \
    }                                                               \
  }

__global__ void fused_forward_kernel(const int m, const int nnz, const int h, const int f,
                                     const int *row_ptr, const int *col_ind, const float *val,
                                     const float *Q, const float *K, const float *V,
                                     float *edge_max, float *edge_sum, float *edge_mask,
                                     float *out_feat, unsigned long long seed)
{
  int rid = blockIdx.x;                     // loop over row of adj matrix
  int hid = blockIdx.y;                     // loop over heads
  int fid = threadIdx.y * 32 + threadIdx.x; // loop over feature dim

  int lb = row_ptr[rid]; // row rid elements
  int hb = row_ptr[rid + 1];
  int ptr = lb + threadIdx.x; // the neighbor node needed to process

  int threads_x = blockDim.x; // 32
  int threads_y = blockDim.y; // f/32

  int num_neighbor = hb - lb;
  extern __shared__ float smem[];
  float *curr_node_feature = smem;
  float *neigh_nodes_weight = (float *)&curr_node_feature[f];

  float weightMax = -1e38;

  // init the shared memory
  if (fid < f)
  {
    curr_node_feature[fid] = Q[rid * h * f + hid * f + fid];
  }
  for (int j = 0; j < num_neighbor / (threads_x * threads_y) + 1; j++)
  {
    int pid = fid + j * (threads_x * threads_y);
    if (pid < num_neighbor)
    {
      neigh_nodes_weight[pid] = 0;
    }
  }
  __syncthreads();

  // compute the attention weight
  for (int j = 0; j < num_neighbor; j++)
  {
    float weight;
    float weight_partial = 0;
    if (fid < f)
    {

      int cid = col_ind[lb + j];
      weight_partial = curr_node_feature[fid] * K[cid * h * f + hid * f + fid];
    }
    atomicAdd(&neigh_nodes_weight[j], weight_partial);
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
    int pid = ptr + (j << 5); // node need to process in loop j
    float exptmp = 0;
    if (pid < hb)
    {
      float weight = neigh_nodes_weight[pid - lb];
      exptmp = exp(weight - weightMax);
    }
    __syncwarp();
    for (int stride = 16; stride > 0; stride >>= 1)
    {
      float tmp = __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      exptmp += tmp;
    }
    __syncwarp();
    expAll += exptmp;
  }
  __syncthreads();

  // compute the output
  float acc = 0;
  for (int j = 0; j < num_neighbor; j++)
  {
    float weight;
    float attn_val;
    int cid = col_ind[lb + j];
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

void gf_forward(int m, int nnz, int h, int f,
                const int *row_ptr, const int *col_ind, const float *val,
                const float *Q, const float *K, const float *V,
                float *edge_max, float *edge_sum, float *edge_mask,
                float *out_feat)
{
  // float rt;
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);

  long seed = clock();
  curandGenerator_t gen;
  (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

  /* Set seed */
  (curandSetPseudoRandomGeneratorSeed(gen, seed));

  const dim3 nblks(m, h, 1);
  const dim3 nthrs(32, (f + 31) / 32, 1);
  printf("start kernel\n");
  // CUDA_KERNEL_CALL(
  //     (fused_forward_kernel),
  //     nblks, nthrs, (f + m) * sizeof(float), m, nnz, h, f, row_ptr, col_ind, val,
  //     Q, K, V, edge_max, edge_sum, edge_mask, out_feat, seed);
  fused_forward_kernel<<<dim3(m, h, 1), dim3(32, (f + 31) / 32, 1),
                         (f + m) * sizeof(float)>>>(
      m, nnz, h, f, row_ptr, col_ind, val,
      Q, K, V, edge_max, edge_sum, edge_mask, out_feat, seed);

  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

std::vector<torch::Tensor>
gf_forward_cuda(torch::Tensor row_ptr,
                torch::Tensor col_ind,
                torch::Tensor val, torch::Tensor Q,
                torch::Tensor K, torch::Tensor V)
{
  // Q: torch.Size([6248, 10, 8])
  const auto m = row_ptr.size(0) - 1; // num nodes
  const auto nnz = col_ind.size(0);   // num edges
  const auto h = Q.size(1);           // num heads
  const auto f = Q.size(2);           // num feats
  auto devid = row_ptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);
  auto edge_max = torch::zeros({m, h}, options);
  auto edge_sum = torch::zeros({m, h}, options);
  auto edge_mask = torch::zeros({nnz, h}, options);
  gf_forward(m, nnz, h, f,
             row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(), val.data_ptr<float>(),
             Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
             edge_max.data_ptr<float>(), edge_sum.data_ptr<float>(), edge_mask.data_ptr<float>(),
             out_feat.data_ptr<float>());
  return {out_feat, edge_max, edge_sum, edge_mask};
}
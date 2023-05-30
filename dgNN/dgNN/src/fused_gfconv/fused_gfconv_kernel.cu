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

__global__ void fused_forward_kernel(const int m, const int nnz, const int h, const int f,
                                     const int *row_ptr, const int *col_ind, const float *val,
                                     const float *Q, const float *K, const float *V,
                                     float *edge_max, float *edge_sum, float *edge_mask,
                                     float *out_feat, unsigned long long seed)
{
  int rid = blockIdx.x;  // loop over row of adj matrix
  int hid = blockIdx.y;  // loop over heads
  int fid = threadIdx.y * 32 + threadIdx.x;; // loop over neighbor nodes in one row
  // int fid = threadIdx.y; // loop over feature dim

  int lb = row_ptr[rid]; // row rid elements
  int hb = row_ptr[rid + 1];
  int ptr = lb + threadIdx.x; // the neighbor node needed to process

  int threads_x = blockDim.x; // 32
  int threads_y = blockDim.y; // f/32

  const int loop = hb - lb;
  extern __shared__ float smem[];
  float *SMEMdata = smem;
  float *curr_node_feature = (float*)&smem[f];
  float *neigh_nodes_weight = (float*)&smem[loop];

  // __shared__ float curr_node_feature[f];
  // __shared__ float neigh_nodes_weight[loop];

  float weight = 0;
  int cid = 0;
  float weightMax = -1e38;
  float expAll = 0;

  for(int j = 0; j <loop;j++){
    float weight_partial;
    cid = col_ind[lb+j];
    weight_partial = curr_node_feature[fid] * V[cid * h * f + hid * f + fid];
    // TODO 应该用原子操作？
    // for (int stride = 16; stride > 0; stride >>= 1)
    // {
    //   float tmp = __shfl_xor_sync(0xffffffff, weight_partial, stride, 32);
    //   weight += tmp;
    // }
    // neigh_nodes_weight[j] = weight;
    atomicAdd(&neigh_nodes_weight[j], weight_partial);
    __syncthreads();
    weight = neigh_nodes_weight[j];
    weightMax = MAX(weight, weightMax);
    
  }

  //   for (int j = 0; j < loop; j++)
  // {
  //   int pid = ptr + (j << 5);
  //   float exptmp = 0;
  //   if (pid < hb)
  //   {
  //     int cid = col_ind[pid];
  //     float attn_col_val = attn_col[cid * h + hid];
  //     float weight = attn_row_val + attn_col_val;
  //     weight = LeakyRelu(weight, negative_slope);
  //     exptmp = exp(weight - weightMax);
  //   }
  //   __syncwarp();
  //   // TODO 这里只用上了一个warp？
  //   for (int stride = 16; stride > 0; stride >>= 1)
  //   {
  //     float tmp = __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
  //     exptmp += tmp;
  //   }
  //   expAll += exptmp;
  // }

  // // compute the sum of each row
  // float expAll = 0;
  // for (int j = 0; j < loop; j++)
  // {
  //   int pid = ptr + (j << 5);
  //   float exptmp = 0;
  //   if (pid < hb)
  //   {
  //     int cid = col_ind[pid];
  //     float attn_col_val = attn_col[cid * h + hid];
  //     float weight = attn_row_val + attn_col_val;
  //     weight = LeakyRelu(weight, negative_slope);
  //     exptmp = exp(weight - weightMax);
  //   }
  //   __syncwarp();
  //   for (int stride = 16; stride > 0; stride >>= 1)
  //   {
  //     float tmp = __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
  //     exptmp += tmp;
  //   }
  //   expAll += exptmp;
  // }
  // // store row sums for backward
  // if (threadIdx.x == 0)
  //   edge_sum[rid * h + hid] = expAll;

  // int fid = threadIdx.y * 32 + threadIdx.x;
  // // for (int fid = threadIdx.x; fid < (f + 31) / 32 * 32; fid += 32)
  // {
  //   float acc = 0;
  //   for (int j = 0; j < loop; j++)
  //   {
  //     int pid = ptr + (j << 5);
  //     float weight = 0;
  //     int cid = 0;
  //     if (pid < hb)
  //     {
  //       cid = col_ind[pid];
  //       weight = 1;
  //       weight = exp(weight - weightMax) / expAll;
  //     }
  //     attn_val_sh[threadIdx.x] = weight;
  //     cid_sh[threadIdx.x] = cid;
  //     __syncwarp();
  //     int jj = lb + (j << 5); // 32 threads process 32 neighbor nodes concurrently
  //     for (int kk = 0; kk < 32 && jj + kk < hb; kk++)
  //     {
  //       int cid = cid_sh[kk];
  //       float val = attn_val_sh[kk];
  //       acc += val * V[cid * h * f + hid * f + fid];
  //     }
  //     __syncwarp();
  //   }
  //   if (fid < f)
  //     out_feat[rid * h * f + hid * f + fid] = acc;
  // }
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

  /* Generate n floats on device */
  (curandGenerateUniform(gen, edge_mask, nnz * h));

  fused_forward_kernel<<<dim3(m, h, 1), dim3(32, (f + 31) / 32, 1),
                         (f+m) * sizeof(float)>>>(
      m, nnz, h, f, row_ptr, col_ind, val,
      Q, K, V, edge_max, edge_sum, edge_mask, out_feat, seed);
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
  auto out_feat = torch::empty({m, h, f}, options);
  auto edge_max = torch::empty({m, h}, options);
  auto edge_sum = torch::empty({m, h}, options);
  auto optionsI =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
  auto edge_mask = torch::empty({nnz, h}, options);
  gf_forward(m, nnz, h, f,
             row_ptr.data_ptr<int>(), col_ind.data_ptr<int>(), val.data_ptr<float>(),
             Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
             edge_max.data_ptr<float>(), edge_sum.data_ptr<float>(), edge_mask.data_ptr<float>(),
             out_feat.data_ptr<float>());
  return {out_feat, edge_max, edge_sum, edge_mask};
}
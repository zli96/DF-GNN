#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType>
__global__ void fused_gt_csr(const int h, const int f, const int *indptr,
                             const int *indices, const DType *val,
                             const DType *Q, const DType *K, const DType *V,
                             DType *out_feat) {
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
  for (int j = 0; j < num_neighbor; j++) {
    int cid = indices[lb + j];
    DType weight = neigh_nodes_weight[j];
    DType attn_val = exp(weight - weightMax);
    if (fid < f) {
      acc += attn_val * V[cid * h * f + hid * f + fid];
    }
  }
  if (fid < f)
    // handle the node with no neighbor
    out_feat[rid * h * f + hid * f + fid] = (expAll != 0) ? acc / expAll : 0;
}

std::vector<torch::Tensor>
gt_csr_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                      torch::Tensor val, int smem_consume, torch::Tensor Q,
                      torch::Tensor K, torch::Tensor V) {
  // Q: torch.Size([6248, 10, 8])
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto nnz = indices.size(0);  // num of edges
  const auto h = Q.size(1);          // num of heads
  const auto f = Q.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);

  const int ntx = WARP_SIZE;
  const int nty = (f + WARP_SIZE - 1) / WARP_SIZE;

  const dim3 nblks(m, h);
  const dim3 nthrs(ntx, nty);

  CUDA_KERNEL_CALL((fused_gt_csr<float>), nblks, nthrs,
                   (smem_consume) * sizeof(float), h, f, indptr.data_ptr<int>(),
                   indices.data_ptr<int>(), val.data_ptr<float>(),
                   Q.data_ptr<float>(), K.data_ptr<float>(),
                   V.data_ptr<float>(), out_feat.data_ptr<float>());
  return {out_feat};
}

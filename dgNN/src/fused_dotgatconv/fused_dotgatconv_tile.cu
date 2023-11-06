#include "../util/computeUtil.h"
#include <cuda.h>
#include <stdio.h>
#include <torch/types.h>
#include <unistd.h>

using namespace std;

template <typename DType>
__global__ void fused_dotgat_tile(const int m, const int h, const int f,
                                  const int *indptr, const int *indices,
                                  const DType *val, const DType *H,
                                  DType *out_feat) {
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
  DType Q_i = H[rid * hf + hfid];

  DType acc = 0, partial_sum = 0;
  DType weightMax_old = -1e38, weightMax = -1e38;
  DType expweight, expweightMax;

  for (int j = 0; j < num_neighbor; j++) {
    DType weight = 0;
    DType weight_partial = 0;

    int cid = indices[lb + j];
    float H_tmp = H[cid * hf + hfid];
    weight_partial = Q_i * H_tmp;
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
    acc = acc * expweightMax + expweight * H_tmp;
    partial_sum = partial_sum * expweightMax + expweight;
    weightMax_old = weightMax;
  }

  // handle the node with no neighbor
  out_feat[rid * hf + hfid] = (partial_sum != 0) ? acc / partial_sum : 0;
}

std::vector<torch::Tensor>
dotgat_tile_forward_cuda(torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor val, int smem_consume, torch::Tensor H) {
  const auto m = indptr.size(0) - 1; // num of nodes
  const auto h = H.size(1);          // num of heads
  const auto f = H.size(2);          // num of feats
  auto devid = indptr.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::zeros({m, h, f}, options);

  const int ntx = WARP_SIZE;
  const int nty = (f + WARP_SIZE - 1) / WARP_SIZE;

  const dim3 nblks(m, h);
  const dim3 nthrs(ntx, nty);
  const int smem_size = smem_consume * sizeof(float);

  CUDA_KERNEL_CALL((fused_dotgat_tile<float>), nblks, nthrs, smem_size, m, h, f,
                   indptr.data_ptr<int>(), indices.data_ptr<int>(),
                   val.data_ptr<float>(), H.data_ptr<float>(),
                   out_feat.data_ptr<float>());
  return {out_feat};
}

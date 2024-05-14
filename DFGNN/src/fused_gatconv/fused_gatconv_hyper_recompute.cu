#include "../util/computeUtil.h"
#include <cuda.h>
#include <torch/types.h>

// template <typename DType>
// __global__ void fused_gat_hyper_recompute_inference(
//     int m, int h, int f, const DType *attn_row, const DType *attn_col,
//     const int *row, const int *indptr, const int *indices, const DType
//     *in_feat, const DType negative_slope, DType *out_feat) {
//   // launch dim (32, 8) * (num_nodes/8, 1)
//   const int bidx = blockIdx.x;
//   const int hid = blockIdx.y;
//   const int tidx = threadIdx.x;
//   const int tidy = threadIdx.y;
//   const int tid = tidy * 32 + tidx;

//   // the node bound of this block
//   const int blockSize = blockDim.y;
//   const int blk_node_lb = blockSize * bidx;
//   const int blk_node_hb = MIN(blk_node_lb + blockSize, m);

//   // the edge bound of this block
//   const int blk_edge_lb = indptr[blk_node_lb];
//   const int blk_edge_hb = indptr[blk_node_hb];

//   // the num of edges in this block
//   const int blk_num_edge = blk_edge_hb - blk_edge_lb;

//   // init smem
//   extern __shared__ DType smem[];
//   DType *neigh_nodes_weight = smem; // [8, f]

//   const int *rowoff = row + blk_edge_lb;
//   const int *indicesoff = indices + blk_edge_lb;

//   // SDDMM, edge parallel
//   for (int i = tid; i < blk_num_edge; i += blockSize * WARP_SIZE) {
//     if (i < blk_num_edge) {
//       const int src = __ldg(rowoff + i);
//       const int dst = __ldg(indicesoff + i);
//       DType weight = attn_row[src * h + hid] + attn_col[dst * h + hid];
//       weight = LeakyRelu(weight, negative_slope);
//       neigh_nodes_weight[i] = weight;
//     }
//   }
//   __syncthreads();

//   // Softmax+SPMM, node parallel
//   int curr_node = blk_node_lb + tidy;
//   if (curr_node < blk_node_hb) {
//     const int edge_lb = indptr[curr_node];
//     const int edge_hb = indptr[curr_node + 1];
//     const int num_edge = edge_hb - edge_lb;

//     DType weightMax = -1e38;
//     const int hf = h * f;
//     // const int hfid = hid * f + tidx;

//     DType *neigh_nodes_weight_off =
//         neigh_nodes_weight + (edge_lb - blk_edge_lb);

//     int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
//     for (int j = 0; j < loop; j++) {
//       DType weight = -1e38;
//       int pid = tidx + (j << 5);
//       if (pid < num_edge) {
//         weight = neigh_nodes_weight_off[pid];
//       }
//       __syncwarp();
// #pragma unroll
//       for (int stride = 16; stride > 0; stride >>= 1) {
//         weight = max(__shfl_xor_sync(0xffffffff, weight, stride, 32),
//         weight);
//       }
//       __syncwarp();
//       weightMax = MAX(weight, weightMax);
//     }

//     // compute the sum of exp
//     DType expAll = 0;
//     for (int j = 0; j < loop; j++) {
//       int pid = tidx + (j << 5); // node need to process in loop j
//       DType exptmp = 0;
//       if (pid < num_edge) {
//         DType weight = neigh_nodes_weight_off[pid];
//         exptmp = exp(weight - weightMax);
//         neigh_nodes_weight_off[pid] = exptmp;
//       }
//       __syncwarp();
// #pragma unroll
//       for (int stride = 16; stride > 0; stride >>= 1) {
//         exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
//       }
//       __syncwarp();
//       expAll += exptmp;
//     }
//     expAll = (expAll != 0) ? 1.0f / expAll : 0;

//     // compute the output
//     int loop_f = (f + WARP_SIZE - 1) / WARP_SIZE;
//     for (int i = 0; i < loop_f; i++) {
//       DType acc = 0;
//       int pid = tidx + (i << 5);
//       for (int j = 0; j < num_edge; j++) {
//         int cid = indices[edge_lb + j];
//         DType attn_val = neigh_nodes_weight_off[j];
//         if (pid < f)
//           acc += attn_val * in_feat[cid * hf + hid * f + pid];
//       }
//       // handle the node with no neighbor
//       if (pid < f)
//         out_feat[curr_node * hf + hid * f + pid] = acc * expAll;
//     }
//   }
// }

template <typename DType>
__global__ void fused_gat_hyper_recompute_inference_vec4(
    int m, int h, int f, const DType *attn_row, const DType *attn_col,
    const int *indptr, const int *indices, const DType *in_feat,
    const DType negative_slope, DType *out_feat) {
  // launch dim (32, 8) * (num_nodes/8, 1)
  const int bidx = blockIdx.x;
  const int hid = blockIdx.y;
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tid = tidy * 32 + tidx;

  // the node bound of this block
  const int blockSize = blockDim.y;
  const int blk_node_lb = blockSize * bidx;
  const int blk_node_hb = MIN(blk_node_lb + blockSize, m);

  // the edge bound of this block
  const int blk_edge_lb = indptr[blk_node_lb];

  // Softmax+SPMM, node parallel
  const int curr_node = blk_node_lb + tidy;
  if (curr_node < blk_node_hb) {
    const int edge_lb = indptr[curr_node];
    const int edge_hb = indptr[curr_node + 1];
    const int num_edge = edge_hb - edge_lb;
    const int *indicesoff = indices + edge_lb;

    DType attn_row_val = attn_row[curr_node * h + hid];
    DType weightMax = -1e38;
    const int hf = h * f;
    int loop = (num_edge + WARP_SIZE - 1) / WARP_SIZE;
    for (int j = 0; j < loop; j++) {
      DType weight = -1e38;
      int pid = tidx + (j << 5);
      if (pid < num_edge) {
        int cid = __ldg(indicesoff + pid);
        DType attn_col_val = attn_col[cid * h + hid];
        weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
      }
      __syncwarp();
#pragma unroll
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
        int cid = __ldg(indicesoff + pid);
        DType attn_col_val = attn_col[cid * h + hid];
        DType weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
        exptmp = exp(weight - weightMax);
      }
      __syncwarp();
#pragma unroll
      for (int stride = 16; stride > 0; stride >>= 1) {
        exptmp += __shfl_xor_sync(0xffffffff, exptmp, stride, 32);
      }
      __syncwarp();
      expAll += exptmp;
    }
    expAll = (expAll != 0) ? 1.0f / expAll : 0;

    DType *Outoff = out_feat + curr_node * h * f + hid * f;
    for (int i = tidx; i < f / 4; i += 32) {
      DType weight = 0;
      DType acc[4] = {0, 0, 0, 0};
      for (int j = 0; j < num_edge; j++) {
        int cid = __ldg(indicesoff + j);
        DType attn_col_val = attn_col[cid * h + hid];
        weight = attn_row_val + attn_col_val;
        weight = LeakyRelu(weight, negative_slope);
        weight = exp(weight - weightMax);
        const DType *featoff = in_feat + cid * h * f + hid * f + 4 * i;
        Mul4_const<float>(acc, featoff, weight);
      }
      // handle the node with no neighbor
      selfMulConst4<float>(acc, expAll);
      Store<float4, float>(Outoff, acc, 4 * i);
    }
  }
}

void gat_hyper_recompute_inference_launch(
    int m, int nnz, int h, int f, const float *attn_row, const float *attn_col,
    const int *indptr, const int *indices, float negative_slope,
    const float *in_feat, float *out_feat) {
  const int ntx = 32;
  const int nty = 8;

  const int nbx = (m + nty - 1) / nty;
  const int nby = h;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  if ((f % 128) == 0) {
    CUDA_KERNEL_CALL((fused_gat_hyper_recompute_inference_vec4<float>), nblks,
                     nthrs, 0, m, h, f, attn_row, attn_col, indptr, indices,
                     in_feat, negative_slope, out_feat);
  } else {
    // TODO: not implemented
    exit(0);
    // CUDA_KERNEL_CALL((fused_gat_hyper_recompute_inference<float>), nblks,
    // nthrs,
    //                  smem_size, m, h, f, attn_row, attn_col, indptr,
    //                  indices, in_feat, negative_slope, out_feat);
  }
}

torch::Tensor gat_hyper_recompute_inference_cuda(
    torch::Tensor attn_row, torch::Tensor attn_col, torch::Tensor indptr,
    torch::Tensor indices, float negative_slope, torch::Tensor in_feat) {
  const auto m = indptr.size(0) - 1;
  const auto nnz = indices.size(0);
  const auto h = attn_row.size(1);
  const auto f = in_feat.size(2);
  auto devid = attn_row.device().index();
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
  auto out_feat = torch::empty({m, h, f}, options);
  gat_hyper_recompute_inference_launch(
      m, nnz, h, f, attn_row.data_ptr<float>(), attn_col.data_ptr<float>(),
      indptr.data_ptr<int>(), indices.data_ptr<int>(), negative_slope,
      in_feat.data_ptr<float>(), out_feat.data_ptr<float>());
  return out_feat;
}
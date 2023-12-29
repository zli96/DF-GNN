#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_DEVICE(x)                                                        \
  TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...)                                                    \
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}),                  \
              #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<torch::Tensor>
gt_csr_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                      torch::Tensor val, int smem_consume, torch::Tensor Q,
                      torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor>
gt_tiling_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor val, int smem_consume, torch::Tensor Q,
                         torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor>
gt_hyper_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                        torch::Tensor rows, torch::Tensor val, int smem_consume,
                        torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor>
gt_softmax_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                          torch::Tensor rows, torch::Tensor val,
                          int smem_consume, torch::Tensor Q, torch::Tensor K,
                          torch::Tensor V);

std::vector<torch::Tensor>
gt_hyper_inference_ablation_cuda(torch::Tensor indptr, torch::Tensor indices,
                                 torch::Tensor rows, torch::Tensor val,
                                 int smem_consume, torch::Tensor Q,
                                 torch::Tensor K, torch::Tensor V);

// std::vector<torch::Tensor>
// gt_subgraph_inference_cuda(torch::Tensor nodes_subgraph, torch::Tensor
// indptr,
//                            torch::Tensor indices, torch::Tensor val,
//                            torch::Tensor Q, torch::Tensor K, torch::Tensor
//                            V);

// std::vector<torch::Tensor> gt_indegree_inference_cuda(
//     torch::Tensor nodes_subgraph, torch::Tensor smem_nodes_subgraph,
//     torch::Tensor store_node, torch::Tensor store_flag, torch::Tensor indptr,
//     torch::Tensor indices, torch::Tensor val, torch::Tensor Q, torch::Tensor
//     K, torch::Tensor V);

// std::vector<torch::Tensor> gt_indegree_hyper_inference_cuda(
//     torch::Tensor nodes_subgraph, torch::Tensor smem_nodes_subgraph,
//     torch::Tensor store_node, torch::Tensor store_flag, torch::Tensor row,
//     torch::Tensor indptr, torch::Tensor indices, torch::Tensor val,
//     torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor>
gt_hyper_forward_cuda(torch::Tensor row_ptr, torch::Tensor col_ind,
                      torch::Tensor rows, torch::Tensor val,
                      torch::Tensor col_ptr, torch::Tensor row_ind,
                      torch::Tensor val_idx, int smem_consume, torch::Tensor Q,
                      torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor>
gt_hyper_forward(torch::Tensor row_ptr, torch::Tensor col_ind,
                 torch::Tensor rows, torch::Tensor val, torch::Tensor col_ptr,
                 torch::Tensor row_ind, torch::Tensor val_idx, int smem_consume,
                 torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // device check
  CHECK_DEVICE(row_ptr);
  CHECK_DEVICE(col_ind);
  CHECK_DEVICE(val);
  CHECK_DEVICE(rows);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(row_ptr);
  CHECK_CONTIGUOUS(col_ind);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(rows);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(rows.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  assert(col_ind.size(0) == val.size(0));

  return gt_hyper_forward_cuda(row_ptr, col_ind, rows, val, col_ptr, row_ind,
                               val_idx, smem_consume, Q, K, V);
}

std::vector<torch::Tensor>
gt_backward_cuda(torch::Tensor row_ptr, torch::Tensor col_ind,
                 torch::Tensor rows, torch::Tensor val, torch::Tensor col_ptr,
                 torch::Tensor row_ind, torch::Tensor val_idx, int smem_consume,
                 torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                 torch::Tensor attn_edge, torch::Tensor grad);

std::vector<torch::Tensor>
gt_backward(torch::Tensor row_ptr, torch::Tensor col_ind, torch::Tensor rows,
            torch::Tensor val, torch::Tensor col_ptr, torch::Tensor row_ind,
            torch::Tensor val_idx, int smem_consume, torch::Tensor Q,
            torch::Tensor K, torch::Tensor V, torch::Tensor attn_edge,
            torch::Tensor grad) {

  CHECK_DEVICE(row_ptr);
  CHECK_DEVICE(col_ind);
  CHECK_DEVICE(rows);
  CHECK_DEVICE(val);
  CHECK_DEVICE(col_ptr);
  CHECK_DEVICE(row_ind);
  CHECK_DEVICE(val_idx);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);
  CHECK_DEVICE(attn_edge);
  CHECK_DEVICE(grad);

  // contiguous check
  CHECK_CONTIGUOUS(row_ptr);
  CHECK_CONTIGUOUS(col_ind);
  CHECK_CONTIGUOUS(rows);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(col_ptr);
  CHECK_CONTIGUOUS(row_ind);
  CHECK_CONTIGUOUS(val_idx);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);
  CHECK_CONTIGUOUS(attn_edge);
  CHECK_CONTIGUOUS(grad);

  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(rows.dtype() == torch::kInt32);
  assert(col_ptr.dtype() == torch::kInt32);
  assert(row_ind.dtype() == torch::kInt32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);
  assert(attn_edge.dtype() == torch::kFloat32);
  assert(grad.dtype() == torch::kFloat32);

  return gt_backward_cuda(row_ptr, col_ind, rows, val, col_ptr, row_ind,
                          val_idx, smem_consume, Q, K, V, attn_edge, grad);
}

std::vector<torch::Tensor> gt_csr_inference(torch::Tensor indptr,
                                            torch::Tensor indices,
                                            torch::Tensor val, int smem_consume,
                                            torch::Tensor Q, torch::Tensor K,
                                            torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  assert(indices.size(0) == val.size(0));

  return gt_csr_inference_cuda(indptr, indices, val, smem_consume, Q, K, V);
}

std::vector<torch::Tensor>
gt_tiling_inference(torch::Tensor indptr, torch::Tensor indices,
                    torch::Tensor val, int smem_consume, torch::Tensor Q,
                    torch::Tensor K, torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  assert(indices.size(0) == val.size(0));

  return gt_tiling_inference_cuda(indptr, indices, val, smem_consume, Q, K, V);
}

std::vector<torch::Tensor>
gt_hyper_inference(torch::Tensor indptr, torch::Tensor indices,
                   torch::Tensor rows, torch::Tensor val, int smem_consume,
                   torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(rows);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(rows);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(rows.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  assert(indices.size(0) == val.size(0));

  return gt_hyper_inference_cuda(indptr, indices, rows, val, smem_consume, Q, K,
                                 V);
}

std::vector<torch::Tensor>
gt_softmax_inference(torch::Tensor indptr, torch::Tensor indices,
                     torch::Tensor rows, torch::Tensor val, int smem_consume,
                     torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(rows);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(rows);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(rows.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  assert(indices.size(0) == val.size(0));

  return gt_softmax_inference_cuda(indptr, indices, rows, val, smem_consume, Q,
                                   K, V);
}

// std::vector<torch::Tensor>
// gt_subgraph_inference(torch::Tensor nodes_subgraph, torch::Tensor indptr,
//                       torch::Tensor indices, torch::Tensor val, torch::Tensor
//                       Q, torch::Tensor K, torch::Tensor V) {
//   // device check
//   CHECK_DEVICE(indptr);
//   CHECK_DEVICE(indices);
//   CHECK_DEVICE(val);
//   CHECK_DEVICE(nodes_subgraph);
//   CHECK_DEVICE(Q);
//   CHECK_DEVICE(K);
//   CHECK_DEVICE(V);

//   // contiguous check
//   CHECK_CONTIGUOUS(indptr);
//   CHECK_CONTIGUOUS(indices);
//   CHECK_CONTIGUOUS(val);
//   CHECK_CONTIGUOUS(nodes_subgraph);
//   CHECK_CONTIGUOUS(Q);
//   CHECK_CONTIGUOUS(K);
//   CHECK_CONTIGUOUS(V);

//   // dtype check
//   assert(indptr.dtype() == torch::kInt32);
//   assert(indices.dtype() == torch::kInt32);
//   assert(nodes_subgraph.dtype() == torch::kInt32);
//   assert(val.dtype() == torch::kFloat32);
//   assert(Q.dtype() == torch::kFloat32);
//   assert(K.dtype() == torch::kFloat32);
//   assert(V.dtype() == torch::kFloat32);

//   // shape check
//   // TODO add shape check
//   assert(indices.size(0) == val.size(0));

//   return gt_subgraph_inference_cuda(nodes_subgraph, indptr, indices, val, Q,
//   K,
//                                     V);
// }

// std::vector<torch::Tensor>
// gt_indegree_inference(torch::Tensor indptr, torch::Tensor indices,
//                       torch::Tensor val, torch::Tensor nodes_subgraph,
//                       torch::Tensor smem_nodes_subgraph,
//                       torch::Tensor store_node, torch::Tensor store_flag,
//                       torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
//   // device check
//   CHECK_DEVICE(indptr);
//   CHECK_DEVICE(indices);
//   CHECK_DEVICE(val);
//   CHECK_DEVICE(nodes_subgraph);
//   CHECK_DEVICE(smem_nodes_subgraph);
//   CHECK_DEVICE(store_node);
//   CHECK_DEVICE(store_flag);
//   CHECK_DEVICE(Q);
//   CHECK_DEVICE(K);
//   CHECK_DEVICE(V);

//   // contiguous check
//   CHECK_CONTIGUOUS(indptr);
//   CHECK_CONTIGUOUS(indices);
//   CHECK_CONTIGUOUS(val);
//   CHECK_CONTIGUOUS(nodes_subgraph);
//   CHECK_CONTIGUOUS(smem_nodes_subgraph);
//   CHECK_CONTIGUOUS(store_node);
//   CHECK_CONTIGUOUS(store_flag);
//   CHECK_CONTIGUOUS(Q);
//   CHECK_CONTIGUOUS(K);
//   CHECK_CONTIGUOUS(V);

//   // dtype check
//   assert(indptr.dtype() == torch::kInt32);
//   assert(indices.dtype() == torch::kInt32);
//   assert(nodes_subgraph.dtype() == torch::kInt32);
//   assert(smem_nodes_subgraph.dtype() == torch::kInt32);
//   assert(store_node.dtype() == torch::kInt32);
//   assert(store_flag.dtype() == torch::kInt32);

//   assert(val.dtype() == torch::kFloat32);
//   assert(Q.dtype() == torch::kFloat32);
//   assert(K.dtype() == torch::kFloat32);
//   assert(V.dtype() == torch::kFloat32);

//   // shape check
//   // TODO add shape check
//   assert(indices.size(0) == val.size(0));

//   return gt_indegree_inference_cuda(nodes_subgraph, smem_nodes_subgraph,
//                                     store_node, store_flag, indptr, indices,
//                                     val, Q, K, V);
// }

// std::vector<torch::Tensor>
// gt_indegree_hyper_inference(torch::Tensor row, torch::Tensor indptr,
//                             torch::Tensor indices, torch::Tensor val,
//                             torch::Tensor nodes_subgraph,
//                             torch::Tensor smem_nodes_subgraph,
//                             torch::Tensor store_node, torch::Tensor
//                             store_flag, torch::Tensor Q, torch::Tensor K,
//                             torch::Tensor V) {
//   // device check
//   CHECK_DEVICE(indptr);
//   CHECK_DEVICE(indices);
//   CHECK_DEVICE(val);
//   CHECK_DEVICE(nodes_subgraph);
//   CHECK_DEVICE(smem_nodes_subgraph);
//   CHECK_DEVICE(store_node);
//   CHECK_DEVICE(store_flag);
//   CHECK_DEVICE(Q);
//   CHECK_DEVICE(K);
//   CHECK_DEVICE(V);

//   // contiguous check
//   CHECK_CONTIGUOUS(indptr);
//   CHECK_CONTIGUOUS(indices);
//   CHECK_CONTIGUOUS(val);
//   CHECK_CONTIGUOUS(nodes_subgraph);
//   CHECK_CONTIGUOUS(smem_nodes_subgraph);
//   CHECK_CONTIGUOUS(store_node);
//   CHECK_CONTIGUOUS(store_flag);
//   CHECK_CONTIGUOUS(Q);
//   CHECK_CONTIGUOUS(K);
//   CHECK_CONTIGUOUS(V);

//   // dtype check
//   assert(indptr.dtype() == torch::kInt32);
//   assert(indices.dtype() == torch::kInt32);
//   assert(nodes_subgraph.dtype() == torch::kInt32);
//   assert(smem_nodes_subgraph.dtype() == torch::kInt32);
//   assert(store_node.dtype() == torch::kInt32);
//   assert(store_flag.dtype() == torch::kInt32);

//   assert(val.dtype() == torch::kFloat32);
//   assert(Q.dtype() == torch::kFloat32);
//   assert(K.dtype() == torch::kFloat32);
//   assert(V.dtype() == torch::kFloat32);

//   // shape check
//   // TODO add shape check
//   assert(indices.size(0) == val.size(0));

//   return gt_indegree_hyper_inference_cuda(nodes_subgraph,
//   smem_nodes_subgraph,
//                                           store_node, store_flag, row,
//                                           indptr, indices, val, Q, K, V);
// }

std::vector<torch::Tensor>
gt_hyper_inference_ablation(torch::Tensor indptr, torch::Tensor indices,
                            torch::Tensor rows, torch::Tensor val,
                            int smem_consume, torch::Tensor Q, torch::Tensor K,
                            torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(rows);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(rows);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(rows.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  assert(indices.size(0) == val.size(0));

  return gt_hyper_inference_ablation_cuda(indptr, indices, rows, val,
                                          smem_consume, Q, K, V);
}

PYBIND11_MODULE(fused_gtconv, m) {
  m.doc() = "fuse sparse ops in graph transformer into one kernel.";
  m.def("gt_hyper_forward", &gt_hyper_forward,
        "fused graph transformer forward op in hyper format, one kernel");
  m.def("gt_backward", &gt_backward,
        "fused graph transformer forward op in hyper format, one kernel");
  m.def("gt_csr_inference", &gt_csr_inference,
        "fused graph transformer inference op");
  m.def("gt_tiling_inference", &gt_tiling_inference,
        "fused graph transformer inference op in tiling method, one kernel");
  m.def("gt_hyper_inference", &gt_hyper_inference,
        "fused graph transformer inference op in hyper format, one kernel");
  m.def("gt_softmax_inference", &gt_softmax_inference,
        "fused graph transformer inference op in softmax format, two kernels");
  m.def("gt_hyper_inference_ablation", &gt_hyper_inference_ablation);
  //   m.def("gt_subgraph_inference", &gt_subgraph_inference,
  //         "fused graph transformer inference op by subgraph");
  //   m.def("gt_indegree_inference", &gt_indegree_inference,
  //         "fused graph transformer inference op by indegree");
  //   m.def("gt_indegree_hyper_inference", &gt_indegree_hyper_inference,
  //         "fused graph transformer inference op by indegree");
}
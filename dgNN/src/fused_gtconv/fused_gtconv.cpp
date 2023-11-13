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
gt_hyper_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                        torch::Tensor rows, torch::Tensor val, int smem_consume,
                        torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor>
gt_softmax_inference_cuda(torch::Tensor indptr, torch::Tensor indices,
                          torch::Tensor rows, torch::Tensor val,
                          int smem_consume, torch::Tensor Q, torch::Tensor K,
                          torch::Tensor V);

std::vector<torch::Tensor>
gt_subgraph_inference_cuda(torch::Tensor nodes_subgraph, torch::Tensor indptr,
                           torch::Tensor indices, torch::Tensor val,
                           torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor> gt_indegree_inference_cuda(
    torch::Tensor nodes_subgraph, torch::Tensor smem_nodes_subgraph,
    torch::Tensor store_node, torch::Tensor store_flag, torch::Tensor indptr,
    torch::Tensor indices, torch::Tensor val, torch::Tensor Q, torch::Tensor K,
    torch::Tensor V);

std::vector<torch::Tensor> gt_indegree_hyper_inference_cuda(
    torch::Tensor nodes_subgraph, torch::Tensor smem_nodes_subgraph,
    torch::Tensor store_node, torch::Tensor store_flag, torch::Tensor row,
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor val,
    torch::Tensor Q, torch::Tensor K, torch::Tensor V);

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
  // TODO ADD TYPE check
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
  // TODO add shape check
  assert(col_ind.size(0) == val.size(0));

  return gt_hyper_forward_cuda(row_ptr, col_ind, rows, val, col_ptr, row_ind,
                               val_idx, smem_consume, Q, K, V);
}

std::vector<torch::Tensor> gt_backward_cuda(
    float negative_slope, float attn_drop, torch::Tensor row_ptr,
    torch::Tensor col_ind, torch::Tensor col_ptr, torch::Tensor row_ind,
    torch::Tensor permute, torch::Tensor edge_max, torch::Tensor edge_sum,
    torch::Tensor edge_mask, torch::Tensor in_feat, torch::Tensor attn_row,
    torch::Tensor attn_col, torch::Tensor grad);

std::vector<torch::Tensor>
gt_backward(float negative_slope, float attn_drop, torch::Tensor row_ptr,
            torch::Tensor col_ind, torch::Tensor col_ptr, torch::Tensor row_ind,
            torch::Tensor permute, torch::Tensor edge_max,
            torch::Tensor edge_sum, torch::Tensor edge_mask,
            torch::Tensor in_feat, torch::Tensor attn_row,
            torch::Tensor attn_col, torch::Tensor grad) {
  assert(row_ptr.device().type() == torch::kCUDA);
  assert(col_ind.device().type() == torch::kCUDA);
  assert(col_ptr.device().type() == torch::kCUDA);
  assert(row_ind.device().type() == torch::kCUDA);
  assert(permute.device().type() == torch::kCUDA);
  // assert(permute2.device().type() == torch::kCUDA);
  // assert(edge_softmax_csr.device().type() == torch::kCUDA);
  // assert(edge_relu_csr.device().type() == torch::kCUDA);
  assert(edge_max.device().type() == torch::kCUDA);
  assert(edge_sum.device().type() == torch::kCUDA);
  assert(edge_mask.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(grad.device().type() == torch::kCUDA);

  assert(row_ptr.is_contiguous());
  assert(col_ind.is_contiguous());
  assert(col_ptr.is_contiguous());
  assert(row_ind.is_contiguous());
  assert(permute.is_contiguous());
  // assert(permute2.is_contiguous());
  // assert(edge_softmax_csr.is_contiguous());
  // assert(edge_relu_csr.is_contiguous());
  assert(edge_max.is_contiguous());
  assert(edge_sum.is_contiguous());
  assert(edge_mask.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(grad.is_contiguous());

  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(col_ptr.dtype() == torch::kInt32);
  assert(row_ind.dtype() == torch::kInt32);
  assert(permute.dtype() == torch::kInt32);
  // assert(permute2.dtype() == torch::kInt32);
  // assert(edge_softmax_csr.dtype() == torch::kFloat32);
  // assert(edge_relu_csr.dtype() == torch::kFloat32);
  assert(edge_max.dtype() == torch::kFloat32);
  assert(edge_sum.dtype() == torch::kFloat32);
  assert(edge_mask.dtype() == torch::kFloat32);
  assert(in_feat.dtype() == torch::kFloat32);
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(grad.dtype() == torch::kFloat32);
  // printf("gat backward\n");

  return gt_backward_cuda(negative_slope, attn_drop, row_ptr, col_ind, col_ptr,
                          row_ind, permute, edge_max, edge_sum, edge_mask,
                          in_feat, attn_row, attn_col, grad);
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
  // TODO add shape check
  assert(indices.size(0) == val.size(0));

  return gt_csr_inference_cuda(indptr, indices, val, smem_consume, Q, K, V);
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
  // TODO add shape check
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
  // TODO add shape check
  assert(indices.size(0) == val.size(0));

  return gt_softmax_inference_cuda(indptr, indices, rows, val, smem_consume, Q,
                                   K, V);
}

std::vector<torch::Tensor>
gt_subgraph_inference(torch::Tensor nodes_subgraph, torch::Tensor indptr,
                      torch::Tensor indices, torch::Tensor val, torch::Tensor Q,
                      torch::Tensor K, torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(nodes_subgraph);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(nodes_subgraph);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(nodes_subgraph.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  // TODO add shape check
  assert(indices.size(0) == val.size(0));

  return gt_subgraph_inference_cuda(nodes_subgraph, indptr, indices, val, Q, K,
                                    V);
}

std::vector<torch::Tensor>
gt_indegree_inference(torch::Tensor indptr, torch::Tensor indices,
                      torch::Tensor val, torch::Tensor nodes_subgraph,
                      torch::Tensor smem_nodes_subgraph,
                      torch::Tensor store_node, torch::Tensor store_flag,
                      torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(nodes_subgraph);
  CHECK_DEVICE(smem_nodes_subgraph);
  CHECK_DEVICE(store_node);
  CHECK_DEVICE(store_flag);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(nodes_subgraph);
  CHECK_CONTIGUOUS(smem_nodes_subgraph);
  CHECK_CONTIGUOUS(store_node);
  CHECK_CONTIGUOUS(store_flag);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(nodes_subgraph.dtype() == torch::kInt32);
  assert(smem_nodes_subgraph.dtype() == torch::kInt32);
  assert(store_node.dtype() == torch::kInt32);
  assert(store_flag.dtype() == torch::kInt32);

  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  // TODO add shape check
  assert(indices.size(0) == val.size(0));

  return gt_indegree_inference_cuda(nodes_subgraph, smem_nodes_subgraph,
                                    store_node, store_flag, indptr, indices,
                                    val, Q, K, V);
}

std::vector<torch::Tensor>
gt_indegree_hyper_inference(torch::Tensor row, torch::Tensor indptr,
                            torch::Tensor indices, torch::Tensor val,
                            torch::Tensor nodes_subgraph,
                            torch::Tensor smem_nodes_subgraph,
                            torch::Tensor store_node, torch::Tensor store_flag,
                            torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(nodes_subgraph);
  CHECK_DEVICE(smem_nodes_subgraph);
  CHECK_DEVICE(store_node);
  CHECK_DEVICE(store_flag);
  CHECK_DEVICE(Q);
  CHECK_DEVICE(K);
  CHECK_DEVICE(V);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(nodes_subgraph);
  CHECK_CONTIGUOUS(smem_nodes_subgraph);
  CHECK_CONTIGUOUS(store_node);
  CHECK_CONTIGUOUS(store_flag);
  CHECK_CONTIGUOUS(Q);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(V);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(nodes_subgraph.dtype() == torch::kInt32);
  assert(smem_nodes_subgraph.dtype() == torch::kInt32);
  assert(store_node.dtype() == torch::kInt32);
  assert(store_flag.dtype() == torch::kInt32);

  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);

  // shape check
  // TODO add shape check
  assert(indices.size(0) == val.size(0));

  return gt_indegree_hyper_inference_cuda(nodes_subgraph, smem_nodes_subgraph,
                                          store_node, store_flag, row, indptr,
                                          indices, val, Q, K, V);
}

PYBIND11_MODULE(fused_gtconv, m) {
  m.doc() = "fuse sparse ops in graph transformer into one kernel. ";
  m.def("gt_hyper_forward", &gt_hyper_forward,
        "fused graph transformer forward op in hyper format, one kernel");
  m.def("gt_backward", &gt_backward,
        "fused graph transformer forward op in hyper format, one kernel");
  m.def("gt_csr_inference", &gt_csr_inference,
        "fused graph transformer inference op");
  m.def("gt_hyper_inference", &gt_hyper_inference,
        "fused graph transformer inference op in hyper format, one kernel");
  m.def("gt_softmax_inference", &gt_softmax_inference,
        "fused graph transformer inference op in hyper format, two kernels");
  m.def("gt_subgraph_inference", &gt_subgraph_inference,
        "fused graph transformer inference op by subgraph");
  m.def("gt_indegree_inference", &gt_indegree_inference,
        "fused graph transformer inference op by indegree");
  m.def("gt_indegree_hyper_inference", &gt_indegree_hyper_inference,
        "fused graph transformer inference op by indegree");
}
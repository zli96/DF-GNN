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

std::vector<torch::Tensor> gf_forward_cuda(torch::Tensor indptr,
                                           torch::Tensor indices,
                                           torch::Tensor val, int smem_consume,
                                           torch::Tensor Q, torch::Tensor K,
                                           torch::Tensor V);

std::vector<torch::Tensor>
gf_ell_forward_cuda(torch::Tensor indptr, torch::Tensor indices,
                    torch::Tensor row_index, torch::Tensor rows_per_tb,
                    torch::Tensor val, torch::Tensor Q, torch::Tensor K,
                    torch::Tensor V);

std::vector<torch::Tensor>
gf_hyper_fused_forward_cuda(torch::Tensor indptr, torch::Tensor indices,
                            torch::Tensor rows, torch::Tensor val,
                            int smem_consume, torch::Tensor Q, torch::Tensor K,
                            torch::Tensor V);

std::vector<torch::Tensor>
gf_hyper_nofuse_forward_cuda(torch::Tensor indptr, torch::Tensor indices,
                             torch::Tensor rows, torch::Tensor val,
                             int smem_consume, torch::Tensor Q, torch::Tensor K,
                             torch::Tensor V);

std::vector<torch::Tensor>
gf_subgraph_forward_cuda(torch::Tensor nodes_subgraph, torch::Tensor indptr,
                         torch::Tensor indices, torch::Tensor val,
                         torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor> gf_outdegree_forward_cuda(
    torch::Tensor nodes_subgraph, torch::Tensor smem_nodes_subgraph,
    torch::Tensor store_node, torch::Tensor store_flag, torch::Tensor indptr,
    torch::Tensor indices, torch::Tensor val, torch::Tensor Q, torch::Tensor K,
    torch::Tensor V);

std::vector<torch::Tensor> gf_forward(torch::Tensor indptr,
                                      torch::Tensor indices, torch::Tensor val,
                                      int smem_consume, torch::Tensor Q,
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
  // TODO add shape check
  assert(indices.size(0) == val.size(0));

  return gf_forward_cuda(indptr, indices, val, smem_consume, Q, K, V);
}

std::vector<torch::Tensor>
gf_hyper_fused_forward(torch::Tensor indptr, torch::Tensor indices,
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

  return gf_hyper_fused_forward_cuda(indptr, indices, rows, val, smem_consume,
                                     Q, K, V);
}

std::vector<torch::Tensor>
gf_hyper_nofuse_forward(torch::Tensor indptr, torch::Tensor indices,
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

  return gf_hyper_nofuse_forward_cuda(indptr, indices, rows, val, smem_consume,
                                      Q, K, V);
}

std::vector<torch::Tensor> gf_ell_forward(torch::Tensor indptr,
                                          torch::Tensor indices,
                                          torch::Tensor row_index,
                                          torch::Tensor rows_per_tb,
                                          torch::Tensor val, torch::Tensor Q,
                                          torch::Tensor K, torch::Tensor V) {
  // device check
  assert(indptr.device().type() == torch::kCUDA);
  assert(indices.device().type() == torch::kCUDA);
  assert(row_index.device().type() == torch::kCUDA);
  assert(rows_per_tb.device().type() == torch::kCUDA);
  assert(val.device().type() == torch::kCUDA);
  assert(Q.device().type() == torch::kCUDA);
  assert(K.device().type() == torch::kCUDA);
  assert(V.device().type() == torch::kCUDA);
  // contiguous check
  assert(indptr.is_contiguous());
  assert(indices.is_contiguous());
  assert(row_index.is_contiguous());
  assert(rows_per_tb.is_contiguous());
  assert(val.is_contiguous());
  assert(Q.is_contiguous());
  assert(K.is_contiguous());
  assert(V.is_contiguous());
  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(row_index.dtype() == torch::kInt32);
  assert(rows_per_tb.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);
  // shape check
  // TODO add shape check
  assert(indices.size(0) == val.size(0));
  return gf_ell_forward_cuda(indptr, indices, row_index, rows_per_tb, val, Q, K,
                             V);
}

std::vector<torch::Tensor>
gf_subgraph_forward(torch::Tensor nodes_subgraph, torch::Tensor indptr,
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

  return gf_subgraph_forward_cuda(nodes_subgraph, indptr, indices, val, Q, K,
                                  V);
}

std::vector<torch::Tensor>
gf_outdegree_forward(torch::Tensor indptr, torch::Tensor indices,
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

  return gf_outdegree_forward_cuda(nodes_subgraph, smem_nodes_subgraph,
                                   store_node, store_flag, indptr, indices, val,
                                   Q, K, V);
}

PYBIND11_MODULE(fused_gfconv, m) {
  m.doc() = "fuse sparse ops in graph transformer into one kernel. ";
  m.def("gf_forward", &gf_forward, "fused graph transformer forward op");
  m.def("gf_ell_forward", &gf_ell_forward,
        "fused graph transformer forward op in ELL format");
  m.def("gf_hyper_fused_forward", &gf_hyper_fused_forward,
        "fused graph transformer forward op in hyper format, one kernel");
  m.def("gf_hyper_nofuse_forward", &gf_hyper_nofuse_forward,
        "fused graph transformer forward op in hyper format, two kernels");
  m.def("gf_subgraph_forward", &gf_subgraph_forward,
        "fused graph transformer forward op by subgraph");
  m.def("gf_outdegree_forward", &gf_outdegree_forward,
        "fused graph transformer forward op by outdegree");
}
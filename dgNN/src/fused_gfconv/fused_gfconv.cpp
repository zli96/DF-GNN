#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> gf_forward_cuda(torch::Tensor row_ptr,
                                           torch::Tensor col_ind,
                                           torch::Tensor val, torch::Tensor Q,
                                           torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor> gf_ell_forward_cuda(torch::Tensor row_ptr,
                                               torch::Tensor col_ind,
                                               torch::Tensor row_index,
                                               torch::Tensor rows_per_tb,
                                               torch::Tensor val, torch::Tensor Q,
                                               torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor> gf_forward(torch::Tensor row_ptr,
                                      torch::Tensor col_ind, torch::Tensor val,
                                      torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V)
{
  // device check
  assert(row_ptr.device().type() == torch::kCUDA);
  assert(col_ind.device().type() == torch::kCUDA);
  assert(val.device().type() == torch::kCUDA);
  assert(Q.device().type() == torch::kCUDA);
  assert(K.device().type() == torch::kCUDA);
  assert(V.device().type() == torch::kCUDA);
  // contiguous check
  assert(row_ptr.is_contiguous());
  assert(col_ind.is_contiguous());
  assert(val.is_contiguous());
  assert(Q.is_contiguous());
  assert(K.is_contiguous());
  assert(V.is_contiguous());
  // dtype check
  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);
  // shape check
  // TODO add shape check
  assert(col_ind.size(0) == val.size(0));
  return gf_forward_cuda(row_ptr, col_ind, val, Q, K, V);
}

std::vector<torch::Tensor> gf_ell_forward(torch::Tensor row_ptr,
                                          torch::Tensor col_ind,
                                          torch::Tensor row_index,
                                          torch::Tensor rows_per_tb,
                                          torch::Tensor val,
                                          torch::Tensor Q, torch::Tensor K,
                                          torch::Tensor V)
{
  // device check
  assert(row_ptr.device().type() == torch::kCUDA);
  assert(col_ind.device().type() == torch::kCUDA);
  assert(row_index.device().type() == torch::kCUDA);
  assert(rows_per_tb.device().type() == torch::kCUDA);
  assert(val.device().type() == torch::kCUDA);
  assert(Q.device().type() == torch::kCUDA);
  assert(K.device().type() == torch::kCUDA);
  assert(V.device().type() == torch::kCUDA);
  // contiguous check
  assert(row_ptr.is_contiguous());
  assert(col_ind.is_contiguous());
  assert(row_index.is_contiguous());
  assert(rows_per_tb.is_contiguous());
  assert(val.is_contiguous());
  assert(Q.is_contiguous());
  assert(K.is_contiguous());
  assert(V.is_contiguous());
  // dtype check
  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(row_index.dtype() == torch::kInt32);
  assert(rows_per_tb.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(Q.dtype() == torch::kFloat32);
  assert(K.dtype() == torch::kFloat32);
  assert(V.dtype() == torch::kFloat32);
  // shape check
  // TODO add shape check
  assert(col_ind.size(0) == val.size(0));
  return gf_ell_forward_cuda(row_ptr, col_ind, row_index, rows_per_tb, val, Q, K, V);
}

PYBIND11_MODULE(fused_gfconv, m)
{
  m.doc() = "fuse sparse ops in graph transformer into one kernel. ";
  m.def("gf_forward", &gf_forward, "fused graph transformer forward op");
  m.def("gf_ell_forward", &gf_ell_forward, "fused graph transformer forward op in ELL format");
}
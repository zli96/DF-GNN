#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

void assertTensor(torch::Tensor &T, torch::ScalarType type) {
  assert(T.is_contiguous());
  assert(T.device().type() == torch::kCUDA);
  assert(T.dtype() == type);
}

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal);

std::vector<torch::Tensor> csr2csc(torch::Tensor rowptr, torch::Tensor colind,
                                   torch::Tensor csr_data) {
  assert(rowptr.device().type() == torch::kCUDA);
  assert(colind.device().type() == torch::kCUDA);
  assert(csr_data.device().type() == torch::kCUDA);
  assert(rowptr.is_contiguous());
  assert(colind.is_contiguous());
  assert(csr_data.is_contiguous());
  assert(rowptr.dtype() == torch::kInt32);
  assert(colind.dtype() == torch::kInt32);
  assert(csr_data.dtype() == torch::kFloat32);
  return csr2csc_cuda(rowptr, colind, csr_data);
}

std::vector<std::vector<std::vector<torch::Tensor>>>
BucketPartHyb(int num_rows, int num_cols, torch::Tensor indptr,
              torch::Tensor indices, int num_col_parts,
              std::vector<int> buckets);

std::vector<std::vector<std::vector<torch::Tensor>>>
csr2ell(int num_rows, int num_cols, torch::Tensor indptr, torch::Tensor indices,
        int num_col_parts, std::vector<int> buckets) {
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(indptr.device().type() == torch::kCPU);
  assert(indices.device().type() == torch::kCPU);
  return BucketPartHyb(num_rows, num_cols, indptr, indices, num_col_parts,
                       buckets);
}

torch::Tensor coo2csr_cuda(torch::Tensor cooRowInd, int m);

torch::Tensor coo2csr(torch::Tensor cooRowInd, int m) {
  assertTensor(cooRowInd, torch::kInt32);
  return coo2csr_cuda(cooRowInd, m);
}

PYBIND11_MODULE(format_conversion, m) {
  m.doc() = "matrix format transformation";
  m.def("csr2csc", &csr2csc, "csr2csc");
  m.def("coo2csr", &coo2csr, "coo2csr");
  m.def("csr2ell", &csr2ell, "csr2ell");
}
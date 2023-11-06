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
dotgat_tile_forward_cuda(torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor val, int smem_consume, torch::Tensor H);

std::vector<torch::Tensor>
dotgat_tile_forward(torch::Tensor indptr, torch::Tensor indices,
                    torch::Tensor val, int smem_consume, torch::Tensor H) {
  // device check
  CHECK_DEVICE(indptr);
  CHECK_DEVICE(indices);
  CHECK_DEVICE(val);
  CHECK_DEVICE(H);

  // contiguous check
  CHECK_CONTIGUOUS(indptr);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(val);
  CHECK_CONTIGUOUS(H);

  // dtype check
  assert(indptr.dtype() == torch::kInt32);
  assert(indices.dtype() == torch::kInt32);
  assert(val.dtype() == torch::kFloat32);
  assert(H.dtype() == torch::kFloat32);

  // shape check
  // TODO add shape check
  assert(indices.size(0) == val.size(0));

  return dotgat_tile_forward_cuda(indptr, indices, val, smem_consume, H);
}

PYBIND11_MODULE(fused_dotgatconv, m) {
  m.doc() = "fuse sparse ops in dot gat into one kernel. ";
  m.def("dotgat_tile_forward", &dotgat_tile_forward,
        "fused graph transformer forward op");
}
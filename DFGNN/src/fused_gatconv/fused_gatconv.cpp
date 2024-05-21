#include <iostream>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor>
gat_forward_cuda(torch::Tensor attn_row, torch::Tensor attn_col,
                 torch::Tensor row_ptr, torch::Tensor col_ind,
                 float negative_slope, torch::Tensor in_feat, float attn_drop);

std::vector<torch::Tensor>
gat_forward(torch::Tensor attn_row, torch::Tensor attn_col,
            torch::Tensor row_ptr, torch::Tensor col_ind, float negative_slope,
            torch::Tensor in_feat, float attn_drop) {
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(row_ptr.device().type() == torch::kCUDA);
  assert(col_ind.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(row_ptr.is_contiguous());
  assert(col_ind.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(in_feat.dtype() == torch::kFloat32);
  return gat_forward_cuda(attn_row, attn_col, row_ptr, col_ind, negative_slope,
                          in_feat, attn_drop);
}

torch::Tensor
gat_softmax_inference_cuda(int smem_consume, torch::Tensor attn_row,
                           torch::Tensor attn_col, torch::Tensor indptr,
                           torch::Tensor indices, torch::Tensor rows,
                           float negative_slope, torch::Tensor in_feat);

torch::Tensor gat_inference_softmax(int smem_consume, torch::Tensor attn_row,
                                    torch::Tensor attn_col,
                                    torch::Tensor indptr, torch::Tensor indices,
                                    torch::Tensor rows, float negative_slope,
                                    torch::Tensor in_feat)

{
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(rows.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(rows.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(rows.dtype() == torch::kInt32);
  assert(in_feat.dtype() == torch::kFloat32);
  return gat_softmax_inference_cuda(smem_consume, attn_row, attn_col, indptr,
                                    indices, rows, negative_slope, in_feat);
}

torch::Tensor
gat_softmax_gm_inference_cuda(torch::Tensor attn_row, torch::Tensor attn_col,
                              torch::Tensor indptr, torch::Tensor indices,
                              torch::Tensor rows, float negative_slope,
                              torch::Tensor in_feat);

torch::Tensor
gat_inference_softmax_gm(torch::Tensor attn_row, torch::Tensor attn_col,
                         torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor rows, float negative_slope,
                         torch::Tensor in_feat)

{
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(rows.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(rows.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(rows.dtype() == torch::kInt32);
  assert(in_feat.dtype() == torch::kFloat32);
  return gat_softmax_gm_inference_cuda(attn_row, attn_col, indptr, indices,
                                       rows, negative_slope, in_feat);
}

torch::Tensor gat_hyper_inference_cuda(int smem_consume, torch::Tensor attn_row,
                                       torch::Tensor attn_col,
                                       torch::Tensor indptr,
                                       torch::Tensor indices,
                                       torch::Tensor rows, float negative_slope,
                                       torch::Tensor in_feat);

torch::Tensor gat_inference_hyper(int smem_consume, torch::Tensor attn_row,
                                  torch::Tensor attn_col, torch::Tensor indptr,
                                  torch::Tensor indices, torch::Tensor rows,
                                  float negative_slope, torch::Tensor in_feat)

{
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(rows.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(rows.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(rows.dtype() == torch::kInt32);
  assert(in_feat.dtype() == torch::kFloat32);
  return gat_hyper_inference_cuda(smem_consume, attn_row, attn_col, indptr,
                                  indices, rows, negative_slope, in_feat);
}

torch::Tensor gat_hyper_recompute_inference_cuda(
    torch::Tensor attn_row, torch::Tensor attn_col, torch::Tensor indptr,
    torch::Tensor indices, float negative_slope, torch::Tensor in_feat);

torch::Tensor
gat_inference_hyper_recompute(torch::Tensor attn_row, torch::Tensor attn_col,
                              torch::Tensor indptr, torch::Tensor indices,
                              float negative_slope, torch::Tensor in_feat)

{
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(in_feat.dtype() == torch::kFloat32);
  return gat_hyper_recompute_inference_cuda(attn_row, attn_col, indptr, indices,
                                            negative_slope, in_feat);
}

torch::Tensor gat_hyper_v2_inference_cuda(int smem_consume, torch::Tensor a_l,
                                          torch::Tensor a_r,
                                          torch::Tensor indptr,
                                          torch::Tensor indices,
                                          float negative_slope,
                                          torch::Tensor in_feat);

torch::Tensor
gat_inference_hyper_v2(int smem_consume, torch::Tensor a_l, torch::Tensor a_r,
                       torch::Tensor indptr, torch::Tensor indices,
                       float negative_slope, torch::Tensor in_feat)

{
  return gat_hyper_v2_inference_cuda(smem_consume, a_l, a_r, indptr, indices,
                                     negative_slope, in_feat);
}

torch::Tensor
gat_hyper_ablation_inference_cuda(int smem_consume, torch::Tensor attn_row,
                                  torch::Tensor attn_col, torch::Tensor indptr,
                                  torch::Tensor indices, torch::Tensor rows,
                                  float negative_slope, torch::Tensor in_feat);

torch::Tensor
gat_inference_hyper_ablation(int smem_consume, torch::Tensor attn_row,
                             torch::Tensor attn_col, torch::Tensor indptr,
                             torch::Tensor indices, torch::Tensor rows,
                             float negative_slope, torch::Tensor in_feat)

{
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(row_ptr.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(row_ptr.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(row_ptr.dtype() == torch::kInt32);
  assert(in_feat.dtype() == torch::kFloat32);
  return gat_hyper_ablation_inference_cuda(smem_consume, attn_row, attn_col,
                                           indptr, indices, rows,
                                           negative_slope, in_feat);
}

torch::Tensor
gat_tiling_inference_cuda(torch::Tensor attn_row, torch::Tensor attn_col,
                          torch::Tensor row_ptr, torch::Tensor col_ind,
                          float negative_slope, torch::Tensor in_feat);

torch::Tensor gat_inference_tiling(torch::Tensor attn_row,
                                   torch::Tensor attn_col,
                                   torch::Tensor row_ptr, torch::Tensor col_ind,
                                   float negative_slope, torch::Tensor in_feat)

{
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(row_ptr.device().type() == torch::kCUDA);
  assert(col_ind.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(row_ptr.is_contiguous());
  assert(col_ind.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(in_feat.dtype() == torch::kFloat32);
  return gat_tiling_inference_cuda(attn_row, attn_col, row_ptr, col_ind,
                                   negative_slope, in_feat);
}

torch::Tensor gat_inference_cuda(torch::Tensor attn_row, torch::Tensor attn_col,
                                 torch::Tensor row_ptr, torch::Tensor col_ind,
                                 float negative_slope, torch::Tensor in_feat);

torch::Tensor gat_inference(torch::Tensor attn_row, torch::Tensor attn_col,
                            torch::Tensor row_ptr, torch::Tensor col_ind,
                            float negative_slope, torch::Tensor in_feat)

{
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(row_ptr.device().type() == torch::kCUDA);
  assert(col_ind.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(row_ptr.is_contiguous());
  assert(col_ind.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(in_feat.dtype() == torch::kFloat32);
  return gat_inference_cuda(attn_row, attn_col, row_ptr, col_ind,
                            negative_slope, in_feat);
}

std::vector<torch::Tensor>
gat_forward_tb_cuda(torch::Tensor attn_row, torch::Tensor attn_col,
                    torch::Tensor row_ptr, torch::Tensor col_ind,
                    float negative_slope, torch::Tensor in_feat,
                    torch::Tensor tile_scheduler);

std::vector<torch::Tensor>
gat_forward_tb(torch::Tensor attn_row, torch::Tensor attn_col,
               torch::Tensor row_ptr, torch::Tensor col_ind,
               float negative_slope, torch::Tensor in_feat,
               torch::Tensor tile_scheduler)

{
  assert(attn_row.device().type() == torch::kCUDA);
  assert(attn_col.device().type() == torch::kCUDA);
  assert(row_ptr.device().type() == torch::kCUDA);
  assert(col_ind.device().type() == torch::kCUDA);
  assert(in_feat.device().type() == torch::kCUDA);
  assert(tile_scheduler.device().type() == torch::kCUDA);
  assert(attn_row.is_contiguous());
  assert(attn_col.is_contiguous());
  assert(row_ptr.is_contiguous());
  assert(col_ind.is_contiguous());
  assert(in_feat.is_contiguous());
  assert(tile_scheduler.is_contiguous());
  assert(attn_row.dtype() == torch::kFloat32);
  assert(attn_col.dtype() == torch::kFloat32);
  assert(row_ptr.dtype() == torch::kInt32);
  assert(col_ind.dtype() == torch::kInt32);
  assert(in_feat.dtype() == torch::kFloat32);
  assert(tile_scheduler.dtype() == torch::kInt32);
  return gat_forward_tb_cuda(attn_row, attn_col, row_ptr, col_ind,
                             negative_slope, in_feat, tile_scheduler);
}

std::vector<torch::Tensor> gat_backward_cuda(
    float negative_slope, float attn_drop, torch::Tensor row_ptr,
    torch::Tensor col_ind, torch::Tensor col_ptr, torch::Tensor row_ind,
    torch::Tensor permute, torch::Tensor edge_max, torch::Tensor edge_sum,
    torch::Tensor edge_mask, torch::Tensor in_feat, torch::Tensor attn_row,
    torch::Tensor attn_col, torch::Tensor grad);

std::vector<torch::Tensor>
gat_backward(float negative_slope, float attn_drop, torch::Tensor row_ptr,
             torch::Tensor col_ind, torch::Tensor col_ptr,
             torch::Tensor row_ind, torch::Tensor permute,
             // torch::Tensor edge_softmax_csr,
             // torch::Tensor edge_relu_csr,
             torch::Tensor edge_max, torch::Tensor edge_sum,
             torch::Tensor edge_mask, torch::Tensor in_feat,
             torch::Tensor attn_row, torch::Tensor attn_col,
             torch::Tensor grad) {
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

  return gat_backward_cuda(negative_slope, attn_drop, row_ptr, col_ind, col_ptr,
                           row_ind, permute, edge_max, edge_sum, edge_mask,
                           in_feat, attn_row, attn_col, grad);
}

PYBIND11_MODULE(fused_gatconv, m) {
  m.doc() = "fuse sparse ops in gat into one kernel. ";
  m.def("gat_forward", &gat_forward, "fused gat forward op");
  m.def("gat_forward_tb", &gat_forward_tb, "fused tb gat forward op");
  m.def("gat_backward", &gat_backward, "fused gat backward op");
  m.def("gat_inference", &gat_inference, "fused gat inference op");
  m.def("gat_inference_hyper", &gat_inference_hyper, "fused gat inference op");
  m.def("gat_inference_hyper_v2", &gat_inference_hyper_v2,
        "fused gat inference op");
  m.def("gat_inference_hyper_recompute", &gat_inference_hyper_recompute,
        "fused gat inference op");
  m.def("gat_inference_softmax", &gat_inference_softmax,
        "fused gat inference op");
  m.def("gat_inference_softmax_gm", &gat_inference_softmax_gm,
        "fused gat inference op");
  m.def("gat_inference_tiling", &gat_inference_tiling);
  m.def("gat_inference_hyper_ablation", &gat_inference_hyper_ablation);
}

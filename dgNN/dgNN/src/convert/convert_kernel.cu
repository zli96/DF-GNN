#include "../util/computeUtil.h"
#include <cuda.h>
#include <cusparse.h>
#include <torch/types.h>
#include <stdexcept>
#include <limits>
#include <iostream>

void csr2cscKernel(int m, int n, int nnz, int *csrRowPtr, int *csrColInd,
                   float *csrVal, int *cscColPtr, int *cscRowInd,
                   float *cscVal)
{
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle));
    size_t bufferSize = 0;
    void *buffer = NULL;
    checkCuSparseError(cusparseCsr2cscEx2_bufferSize(
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
        cscRowInd, CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, &bufferSize));
    checkCudaError(cudaMalloc((void **)&buffer, bufferSize * sizeof(float)));
    checkCuSparseError(cusparseCsr2cscEx2(
        handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
        cscRowInd, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1, buffer));
    checkCudaError(cudaFree(buffer));
}

std::vector<torch::Tensor> csr2csc_cuda(torch::Tensor csrRowPtr,
                                        torch::Tensor csrColInd,
                                        torch::Tensor csrVal)
{
    const auto n = csrRowPtr.size(0) - 1;
    const auto nnz = csrColInd.size(0);
    auto devid = csrRowPtr.device().index();
    auto optionsF =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto optionsI =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
    auto cscColPtr = torch::empty({n + 1}, optionsI);
    auto cscRowInd = torch::empty({nnz}, optionsI);
    auto cscVal = torch::empty({nnz}, optionsF);
    csr2cscKernel(n, n, nnz, csrRowPtr.data_ptr<int>(), csrColInd.data_ptr<int>(),
                  csrVal.data_ptr<float>(), cscColPtr.data_ptr<int>(),
                  cscRowInd.data_ptr<int>(), cscVal.data_ptr<float>());
    return {cscColPtr, cscRowInd, cscVal};
}

void coo2csrKernel(int m, int nnz, int *cooRowInd, int *csrRowPtr)
{
    cusparseHandle_t handle;
    checkCuSparseError(cusparseCreate(&handle));
    checkCuSparseError(cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, CUSPARSE_INDEX_BASE_ZERO));
}

torch::Tensor coo2csr_cuda(torch::Tensor cooRowInd, int m)
{
    const auto nnz = cooRowInd.size(0);
    auto devid = cooRowInd.device().index();
    auto optionsI = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, devid);
    auto csrRowPtr = torch::empty({m + 1}, optionsI);
    coo2csrKernel(m, nnz, cooRowInd.data_ptr<int>(), csrRowPtr.data_ptr<int>());
    return csrRowPtr;
}

std::vector<std::vector<std::vector<torch::Tensor>>> BucketPartHyb(int num_rows, int num_cols, torch::Tensor indptr,
                                                                   torch::Tensor indices, int num_col_parts,
                                                                   std::vector<int> buckets)
{
    int partition_size = (num_cols + num_col_parts - 1) / num_col_parts;
    int num_bkts = buckets.size();
    std::vector<int> buckets_vec;
    for (const int &bucket_size : buckets)
    {
        buckets_vec.push_back(bucket_size);
    }

    int *indptr_data = static_cast<int *>(indptr.data_ptr<int>());
    int *indices_data = static_cast<int *>(indices.data_ptr<int>());
    std::vector<std::unordered_multiset<int>> degree_counter(num_col_parts);
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j)
        {
            int row_id = i;
            int col_id = indices_data[j];
            int part_id = col_id / partition_size;
            degree_counter[part_id].insert(row_id);
        }
    }

    /* (num_parts, num_buckets, ...) */
    std::vector<std::vector<std::vector<int>>> row_indices(num_col_parts);
    std::vector<std::vector<std::vector<int>>> col_indices(num_col_parts);
    std::vector<std::vector<std::vector<int>>> mask(num_col_parts);
    // init row_indices, col_indices, mask
    for (int part_id = 0; part_id < num_col_parts; ++part_id)
    {
        for (int bucket_id = 0; bucket_id < num_bkts; ++bucket_id)
        {
            row_indices[part_id].push_back(std::vector<int>());
            col_indices[part_id].push_back(std::vector<int>());
            mask[part_id].push_back(std::vector<int>());
        }
    }
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = indptr_data[i]; j < indptr_data[i + 1]; ++j)
        {
            int row_id = i;
            int col_id = indices_data[j];
            int part_id = col_id / partition_size;
            int degree = degree_counter[part_id].count(row_id);
            int bucket_id = std::upper_bound(buckets_vec.begin(), buckets_vec.end(), degree - 1) -
                            buckets_vec.begin();
            if (bucket_id == num_bkts)
            {
                bucket_id--;
            }
            int bucket_size = buckets_vec[bucket_id];
            bool create_new_bucket = false;
            int remainder = col_indices[part_id][bucket_id].size() % bucket_size;
            if (remainder != 0)
            {
                if (row_indices[part_id][bucket_id].empty())
                    throw std::invalid_argument("row indices should not be empty.");
                // if(!row_indices[part_id][bucket_id].empty())
                // std::cout<< "row indices should not be empty.";
                if (row_id != row_indices[part_id][bucket_id].back())
                {
                    // padding
                    for (int k = remainder; k < bucket_size; ++k)
                    {
                        col_indices[part_id][bucket_id].push_back(0);
                        mask[part_id][bucket_id].push_back(0);
                    }
                    create_new_bucket = true;
                }
            }
            else
            {
                create_new_bucket = true;
            }
            if (create_new_bucket)
            {
                // ICHECK(col_indices[part_id][bucket_id].size() % bucket_size == 0) << "Invalid padding";
                if (col_indices[part_id][bucket_id].size() % bucket_size != 0)
                    throw std::invalid_argument("Invalid padding");
                row_indices[part_id][bucket_id].push_back(row_id);
            }
            col_indices[part_id][bucket_id].push_back(col_id);
            mask[part_id][bucket_id].push_back(1);
        }
    }

    // final padding and conversion to torch::Tensor
    std::vector<std::vector<torch::Tensor>> row_indices_nd;
    std::vector<std::vector<torch::Tensor>> col_indices_nd;
    std::vector<std::vector<torch::Tensor>> mask_nd;
    for (int part_id = 0; part_id < num_col_parts; ++part_id)
    {
        std::vector<torch::Tensor> row_indices_part_local;
        std::vector<torch::Tensor> col_indices_part_local;
        std::vector<torch::Tensor> mask_part_local;
        for (int bucket_id = 0; bucket_id < num_bkts; ++bucket_id)
        {
            int bucket_size = buckets_vec[bucket_id];
            // padding
            int remainder = col_indices[part_id][bucket_id].size() % bucket_size;
            if (remainder != 0)
            {
                for (int k = remainder; k < bucket_size; ++k)
                {
                    col_indices[part_id][bucket_id].push_back(0);
                    mask[part_id][bucket_id].push_back(0);
                }
            }
            // conversion to torch::Tensor
            int nnz = row_indices[part_id][bucket_id].size();
            // std::cout << "part&bucket " << part_id << " " << bucket_id << std::endl;
            // std::cout << col_indices[part_id][bucket_id].size() << " " << nnz * bucket_size << std::endl;
            // std::cout << mask[part_id][bucket_id].size() << " " << nnz * bucket_size << std::endl;

            if (static_cast<int>(col_indices[part_id][bucket_id].size()) != nnz * bucket_size)
                    throw std::invalid_argument("Padding error.");
            if (static_cast<int>(mask[part_id][bucket_id].size()) != nnz * bucket_size)
                    throw std::invalid_argument("Padding error.");
            // ICHECK(static_cast<int>(col_indices[part_id][bucket_id].size()) == nnz * bucket_size)
            //     << "Padding error.";
            // ICHECK(static_cast<int>(mask[part_id][bucket_id].size()) == nnz * bucket_size)
            //     << "Padding error.";
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            torch::Tensor row_indices_bucket_local = torch::empty({nnz}, options);
            torch::Tensor col_indices_bucket_local = torch::empty({nnz, bucket_size}, options);
            torch::Tensor mask_bucket_local = torch::empty({nnz, bucket_size}, options);
            if (nnz > 0)
            {
                auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

                row_indices_bucket_local = torch::from_blob(row_indices[part_id][bucket_id].data(),
                                                            {nnz}, opts).clone();
                col_indices_bucket_local = torch::from_blob(col_indices[part_id][bucket_id].data(),
                                                           {nnz, bucket_size}, opts).clone();
                mask_bucket_local = torch::from_blob(mask[part_id][bucket_id].data(),
                                                     {nnz, bucket_size}, opts).clone();
            }
            row_indices_part_local.push_back(row_indices_bucket_local.to(torch::kLong));
            col_indices_part_local.push_back(col_indices_bucket_local.to(torch::kLong));
            mask_part_local.push_back(mask_bucket_local.to(torch::kLong));
        }
        row_indices_nd.push_back(row_indices_part_local);
        col_indices_nd.push_back(col_indices_part_local);
        mask_nd.push_back(mask_part_local);
    }

    return {row_indices_nd, col_indices_nd, mask_nd};
}
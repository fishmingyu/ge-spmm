#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>

torch::Tensor csr2csc_cuda(
    torch::Tensor csrRowPtr,
    torch::Tensor csrColInd,
    torch::Tensor csrVal,
    torch::Tensor cscColPtr,
    torch::Tensor cscRowInd
);

torch::Tensor csr2csc(
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor colptr,
    torch::Tensor rowind,
    torch::Tensor csr_data
) {
    assert(rowptr.device().type()==torch::kCUDA);
    assert(colind.device().type()==torch::kCUDA);
    assert(colptr.device().type()==torch::kCUDA);
    assert(rowind.device().type()==torch::kCUDA);
    assert(csr_data.device().type()==torch::kCUDA);
    assert(rowptr.is_contiguous());
    assert(colind.is_contiguous());
    assert(colptr.is_contiguous());
    assert(rowind.is_contiguous());
    assert(csr_data.is_contiguous());
    assert(rowptr.dtype()==torch::kInt32);
    assert(colind.dtype()==torch::kInt32);
    assert(colptr.dtype()==torch::kInt32);
    assert(rowind.dtype()==torch::kInt32);
    assert(csr_data.dtype()==torch::kFloat32); 
    return csr2csc_cuda(rowptr, colind, csr_data, colptr, rowind);
}

PYBIND11_MODULE(csr2csc, m)
{
    m.doc() = "csr2csc code by using cuSparse library";
    m.def("csr2csc", &csr2csc, "csr2csc");
}
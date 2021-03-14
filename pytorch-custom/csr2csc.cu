#include <cuda.h>
#include <torch/types.h>
#include <cusparse.h>

#define checkCudaError( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while(0)

#define checkCuSparseError( a ) do { \
    if (CUSPARSE_STATUS_SUCCESS != (a)) { \
    fprintf(stderr, "CuSparse runTime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
} while (0)

void csr2cscKernel(int m, int n, int nnz,
    int *csrRowPtr, int *csrColInd, float *csrVal,
    int *cscColPtr, int *cscRowInd, float *cscVal
)
{
    cusparseHandle_t handle;
    size_t bufferSize = 0;
    void* buffer = NULL;
    checkCuSparseError(cusparseCsr2cscEx2_bufferSize(handle,
        m,
        n,
        nnz,
        csrVal,
        csrRowPtr,
        csrColInd,
        cscVal,
        cscColPtr,
        cscRowInd,
        CUDA_R_32F,
        CUSPARSE_ACTION_SYMBOLIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &bufferSize
    ));
    checkCudaError(cudaMalloc((void**)&buffer, bufferSize * sizeof(float)));
    checkCuSparseError(cusparseCsr2cscEx2(handle,
        m,
        n,
        nnz,
        csrVal,
        csrRowPtr,
        csrColInd,
        cscVal,
        cscColPtr,
        cscRowInd,
        CUDA_R_32F,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        buffer
    ));
    checkCudaError(cudaFree(buffer));
}

torch::Tensor csr2csc_cuda(
    torch::Tensor csrRowPtr,
    torch::Tensor csrColInd,
    torch::Tensor csrVal,
    torch::Tensor cscColPtr,
    torch::Tensor cscRowInd
)
{
    const auto m = csrRowPtr.size(0) - 1;
    const auto n = cscColPtr.size(0) - 1;
    const auto nnz = csrColInd.size(0);
    auto devid = csrRowPtr.device().index();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, devid);
    auto cscVal = torch::empty({nnz}, options);
    csr2cscKernel(m, n, nnz, csrRowPtr.data_ptr<int>(), csrColInd.data_ptr<int>(), csrVal.data_ptr<float>(), 
    cscColPtr.data_ptr<int>(), cscRowInd.data_ptr<int>(), cscVal.data_ptr<float>());
    return cscVal;
}


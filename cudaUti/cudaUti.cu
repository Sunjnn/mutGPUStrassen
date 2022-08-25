#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>

#include "cudaUti.cuh"


// basic routine. compute C = AB.
// C, A and B must be consecutive.
// could set the cuda stream.
void gemmcublas(float *C, float *A, float *B, int M, int K, int N, cublasHandle_t handle, cudaStream_t stream) {
    cublasStatus_t cublasStatus = CUBLAS_STATUS_SUCCESS;
    cublasStatus = cublasSetStream(handle, stream);
    CHECKCUBLAS(cublasStatus);

    //void *workspace{ nullptr };
    //CHECKCUDA(cudaMallocAsync(&workspace, 1024 * 1024, stream));
    //CHECKCUBLAS(cublasSetWorkspace(handle, workspace, 1024 * 1024));

    cudaError_t cudaStatus;
    float *d_A{nullptr}, *d_B{nullptr}, *d_C{nullptr};
    cudaStatus = cudaMallocAsync(&d_A, sizeof(float) * M * K, stream);
    CHECKCUDA(cudaStatus);
    cudaStatus = cudaMallocAsync(&d_B, sizeof(float) * K * N, stream);
    CHECKCUDA(cudaStatus);
    cudaStatus = cudaMallocAsync(&d_C, sizeof(float) * M * N, stream);
    CHECKCUDA(cudaStatus);

    cublasStatus = cublasSetMatrixAsync(M, K, sizeof(float), A, M, d_A, M, stream);
    CHECKCUBLAS(cublasStatus);
    cublasStatus = cublasSetMatrixAsync(K, N, sizeof(float), B, K, d_B, K, stream);
    CHECKCUBLAS(cublasStatus);

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, N, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cublasStatus = cublasGetMatrixAsync(M, N, sizeof(float), d_C, M, C, M, stream);
    CHECKCUBLAS(cublasStatus);

    cudaStatus = cudaFreeAsync(d_A, stream);
    CHECKCUDA(cudaStatus);
    cudaStatus = cudaFreeAsync(d_B, stream);
    CHECKCUDA(cudaStatus);
    cudaStatus = cudaFreeAsync(d_C, stream);
    CHECKCUDA(cudaStatus);
}

// basic routine. compute C = AB.
// C, A and B must be consecutive.
void gemmcublas(float *C, float *A, float *B, int M, int K, int N, cublasHandle_t handle) {
    float *d_A{nullptr}, *d_B{nullptr}, *d_C{nullptr};
    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);
    cudaMalloc(&d_C, sizeof(float) * M * N);

    cublasSetMatrix(M, K, sizeof(float), A, M, d_A, M);
    cublasSetMatrix(K, N, sizeof(float), B, K, d_B, K);

    float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    cublasGetMatrix(N, N, sizeof(float), d_C, M, C, M);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// get the number of system.
int getdevicecount() {
    int deviceCount = 0;
    CHECKCUDA(cudaGetDeviceCount(&deviceCount));
    return deviceCount;
}

// get properties of each GPU: memory.
float *getdeviceprop(int deviceCount) {
    if (deviceCount == 0) {
        printf("There are no avilable device(s) that support CUDA\n");
        exit(1);
    }
    else {
        printf("Delete %d CUDA Capable device(s)\n", deviceCount);
    }

    float* memMiBs = (float*)malloc(sizeof(float) * deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        //CHECKCUDA(cudaSetDevice(dev));
        cudaDeviceProp deviceProp;
        CHECKCUDA(cudaGetDeviceProperties(&deviceProp, dev));
        //float memMiB = deviceProp.totalGlobalMem / 1048576.0f;
        memMiBs[dev] = deviceProp.totalGlobalMem / 1048576.0f;
    }

    return memMiBs;
}

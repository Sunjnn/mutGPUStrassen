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

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
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
float getdeviceprop(int dev) {
    float mem;
    cudaDeviceProp deviceProp;
    CHECKCUDA(cudaGetDeviceProperties(&deviceProp, dev));
    mem = deviceProp.totalGlobalMem / 1048576.0f;

    return mem;
}

// block size 64, 8
__global__ void GPU8_mul(float *d_C, int ldC, float *d_A, int ldA, float *d_B, int ldB) {
    __shared__ float as[64][64];
    __shared__ float bs[8][64];

    float cr[8] = {0};

    int blockXTimes64 = blockIdx.x * 64;
    int blockYTimes64 = blockIdx.y * 64;
    int threadYTimes8 = threadIdx.y * 8;

    int cNext = (blockYTimes64 + threadYTimes8) * ldC + blockXTimes64 + threadIdx.x;
    int aNext = threadIdx.y * ldA + blockXTimes64 + threadIdx.x;
    int bNext = (blockYTimes64 + threadYTimes8) * ldB + threadIdx.x;

    d_C += cNext;
    d_A += aNext;
    d_B += bNext;
    float *d_BTmp = d_B;

    int nDiv64 = ldB / 64;
    int ldAtimes8 = ldA * 8;

    for (int i = 0; i < nDiv64; ++i) {
        as[threadIdx.y][threadIdx.x] = d_A[0];
        d_A += ldAtimes8;
        as[threadIdx.y + 8][threadIdx.x] = d_A[0];
        d_A += ldAtimes8;
        as[threadIdx.y + 16][threadIdx.x] = d_A[0];
        d_A += ldAtimes8;
        as[threadIdx.y + 24][threadIdx.x] = d_A[0];
        d_A += ldAtimes8;

        for (int j = 0; j < 8; ++j) {
            bs[threadIdx.y][threadIdx.x] = d_BTmp[0];
            d_BTmp += ldB;
            __syncthreads();
            for (int k = 0; k < 64; ++k) {
                cr[j] = as[k][threadIdx.x] * bs[threadIdx.y][k];
            }
        }

        d_B += 64;
        d_BTmp = d_B;
    }

    for (int i = 0; i < 8; ++i) {
        d_C[i] = cr[i];
    }
}

// block size 32, 32
__global__ void GPU8_add(float *d_C, int ldC, float *d_A, int ldA, float *d_B, int ldB, int M, int N) {
    int y = 32 * blockIdx.y + threadIdx.y;
    int x = 32 * blockIdx.x + threadIdx.x;
    int idxC = y * ldC + x;
    int idxA = y * ldA + x;
    int idxB = y * ldB + x;
    if (x < M && y < N) {
        d_C[idxC] = d_A[idxA] + d_B[idxB];
    }
}

// block size 32, 32
__global__ void GPU8_sub(float *d_C, int ldC, float *d_A, int ldA, float *d_B, int ldB, int M, int N) {
    int y = 32 * blockIdx.y + threadIdx.y;
    int x = 32 * blockIdx.x + threadIdx.x;
    int idxC = y * ldC + x;
    int idxA = y * ldA + x;
    int idxB = y * ldB + x;
    if (x < M && y < N) {
        d_C[idxC] = d_A[idxA] - d_B[idxB];
    }
}
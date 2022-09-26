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

// block size 32, 2
__global__ void GPU8_mul(float *d_C, int ldC, float *d_A, int ldA, float *d_B, int ldB) {
    __shared__ float bs[16][32];

    float c[16] = {0};
    float a;

    int blockXTimes64 = blockIdx.x * 64;
    int blockYTimes16 = blockIdx.y * 16;
    int tid = threadIdx.y * 32 + threadIdx.x;

    int cNext = blockYTimes16 * ldC + blockXTimes64 + tid;
    int aNext = blockXTimes64 + tid;
    int bNext = (blockYTimes16 + threadIdx.y) * ldB + threadIdx.x;

    d_C += cNext;
    d_A += aNext;
    d_B += bNext;
    float *d_BTmp = d_B;

    int nDiv16 = ldB / 32;

    for (int i = 0; i < nDiv16; ++i) {
        bs[threadIdx.y][threadIdx.x] = d_BTmp[0];
        d_BTmp += 2 * ldB;
        bs[threadIdx.y + 2][threadIdx.x] = d_BTmp[0];
        d_BTmp += 2 * ldB;
        bs[threadIdx.y + 4][threadIdx.x] = d_BTmp[0];
        d_BTmp += 2 * ldB;
        bs[threadIdx.y + 6][threadIdx.x] = d_BTmp[0];
        d_BTmp += 2 * ldB;
        bs[threadIdx.y + 8][threadIdx.x] = d_BTmp[0];
        d_BTmp += 2 * ldB;
        bs[threadIdx.y + 10][threadIdx.x] = d_BTmp[0];
        d_BTmp += 2 * ldB;
        bs[threadIdx.y + 12][threadIdx.x] = d_BTmp[0];
        d_BTmp += 2 * ldB;
        bs[threadIdx.y + 14][threadIdx.x] = d_BTmp[0];
        d_BTmp += 2 * ldB;

        for (int j = 0; j < 32; ++j) {
            a = d_A[0];
            c[0] += a * bs[0][j];
            c[1] += a * bs[1][j];
            c[2] += a * bs[2][j];
            c[3] += a * bs[3][j];
            c[4] += a * bs[4][j];
            c[5] += a * bs[5][j];
            c[6] += a * bs[6][j];
            c[7] += a * bs[7][j];
            c[8] += a * bs[8][j];
            c[9] += a * bs[9][j];
            c[10] += a * bs[10][j];
            c[11] += a * bs[11][j];
            c[12] += a * bs[12][j];
            c[13] += a * bs[13][j];
            c[14] += a * bs[14][j];
            c[15] += a * bs[15][j];

            d_A += ldA;
        }

        d_B += 32;
        d_BTmp = d_B;
    }

    for (int i = 0; i < 16; ++i) {
        d_C[0] = c[i];
        d_C += ldC;
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
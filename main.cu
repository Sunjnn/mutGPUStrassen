#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <functional>

#include "main.cuh"
#include "cudaUti.cuh"
#include "gemmStrassen.cuh"
#include "matrixUti.hxx"
#include "blockMatrix.hxx"
#include "threadPool.cuh"


int main() {
    float *A = (float*)malloc(sizeof(float) * M * K);
    float *B = (float*)malloc(sizeof(float) * K * N);
    float *C = (float*)malloc(sizeof(float) * M * N);
    float *CTest = (float*)malloc(sizeof(float) * M * N);

    initMatrix(A, M, K);
    initMatrix(B, K, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;

    cublasHandle_t handle;
    cublasCreate(&handle);
    gemmcublas(CTest, A, B, M, K, N, handle);

    memset(CTest, 0, sizeof(float) * M * N);

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    gemmstrassenNOomp(CTest, A, B, M, K, N, handle);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("gemmstrassenNOomp time: %f ms\n", time);
    // test(C, CTest, M, N);

    memset(C, 0, sizeof(float) * M * N);
    time = 0.0f;

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    gemmstrassen(C, A, B, M, K, N, M, K, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("gemmstrassen time: %f ms\n", time);
    test(C, CTest, M, N);

    memset(C, 0, sizeof(float) * M * N);
    time = 0.0f;

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    // threadPool pool(C, A, B, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N);
    // pool.threadCPU();
    threadPoolConfig config(C, A, B, M, K, N, BLOCK_M, BLOCK_K, BLOCK_N);
    threadCPU(&config);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("gemmstrassen threadPool time: %f ms\n", time);
    test(C, CTest, M, N);

    return 0;
}

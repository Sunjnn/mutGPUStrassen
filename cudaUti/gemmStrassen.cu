#include <cublas_v2.h>

#include <stdio.h>
#include <omp.h>

#include "matrixUti.hxx"
#include "cudaUti.cuh"


// gemm by strassen algorithm. compute C = AB.
// parallel by OpenMP and streams.
// C, A and B must be consecutive.
void gemmstrassen(float *C, float *A, float *B, int m, int k, int n) {
    int mDiv2 = m / 2;
    int kDiv2 = k / 2;
    int nDiv2 = n / 2;

    float *A_11, *A_12, *A_21, *A_22;
    getSubmatrixPointer(A, mDiv2, kDiv2, m, A_11, A_12, A_21, A_22);

    float *B_11, *B_12, *B_21, *B_22;
    getSubmatrixPointer(B, kDiv2, nDiv2, k, B_11, B_12, B_21, B_22);

    float *M_1A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_1B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M1 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_2A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_2B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M2 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_3A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_3B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M3 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_4A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_4B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M4 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_5A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_5B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M5 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_6A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_6B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M6 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_7A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_7B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M7 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);

    cublasHandle_t *handleArray = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * 7);
    for (int i = 0; i < 7; ++i) {
        cublasCreate(handleArray + i);
    }

    cudaStream_t *streamArray = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 7);
    for (int i = 0; i < 7; ++i) {
        cudaStreamCreate(streamArray + i);
    }

    int id;
#pragma omp parallel private(id)
    {
#pragma omp sections
        {
#pragma omp section
            {
                matrixAdd(M_1A, A_11, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_1B, B_11, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handleArray[0], streamArray[0]);
                //gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handleArray[0]);
            }

#pragma omp section
            {
                matrixAdd(M_2A, A_21, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixCopy(M_2B, B_11, kDiv2, nDiv2, kDiv2, k);
                gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handleArray[1], streamArray[1]);
                //gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handleArray[1]);
            }
#pragma omp section
            {
                matrixCopy(M_3A, A_11, mDiv2, kDiv2, mDiv2, m);
                matrixMinus(M_3B, B_12, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handleArray[2], streamArray[2]);
                //gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handleArray[2]);
            }
#pragma omp section
            {
                matrixCopy(M_4A, A_22, mDiv2, kDiv2, mDiv2, m);
                matrixMinus(M_4B, B_21, B_11, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handleArray[3], streamArray[3]);
                //gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handleArray[3]);
            }
#pragma omp section
            {
                matrixAdd(M_5A, A_11, A_12, mDiv2, kDiv2, mDiv2, m, m);
                matrixCopy(M_5B, B_22, kDiv2, nDiv2, kDiv2, k);
                gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handleArray[4], streamArray[4]);
                //gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handleArray[4]);
            }
#pragma omp section
            {
                matrixMinus(M_6A, A_21, A_11, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_6B, B_11, B_12, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handleArray[5], streamArray[5]);
                //gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handleArray[5]);
            }
#pragma omp section
            {
                matrixMinus(M_7A, A_12, A_22, mDiv2, kDiv2, mDiv2, m, m);
                matrixAdd(M_7B, B_21, B_22, kDiv2, nDiv2, kDiv2, k, k);
                gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handleArray[6], streamArray[6]);
                //gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handleArray[6]);
            }
        }
    }

    float *C_11, *C_12, *C_21, *C_22;
    getSubmatrixPointer(C, mDiv2, nDiv2, m, C_11, C_12, C_21, C_22);

#pragma omp parallel
    {
#pragma omp sections
        {
#pragma omp section
            {
                matrixAdd(C_11, M1, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);
                matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, m, m, mDiv2);
                matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, m, m, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_21, M2, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_12, M3, M5, mDiv2, nDiv2, m, mDiv2, mDiv2);
            }
#pragma omp section
            {
                matrixMinus(C_22, M1, M2, mDiv2, nDiv2, m, mDiv2, mDiv2);
                matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, m, m, mDiv2);
                matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, m, m, mDiv2);
            }
        }
    }

    free(M_1A);
    free(M_1B);
    free(M1);
    free(M_2A);
    free(M_2B);
    free(M2);
    free(M_3A);
    free(M_3B);
    free(M3);
    free(M_4A);
    free(M_4B);
    free(M4);
    free(M_5A);
    free(M_5B);
    free(M5);
    free(M_6A);
    free(M_6B);
    free(M6);
    free(M_7A);
    free(M_7B);
    free(M7);

    for (int i = 0; i < 7; ++i) {
        cublasDestroy(handleArray[i]);
        cudaStreamDestroy(streamArray[i]);
    }
    free(handleArray);
    free(streamArray);
}

// gemm by strassen algorithm. compute C = AB.
// parallel by OpenMP and streams.
void gemmstrassen(float *C, float *A, float *B, int m, int k, int n, int ldA, int ldB, int ldC) {
    int mDiv2 = m / 2;
    int kDiv2 = k / 2;
    int nDiv2 = n / 2;

    float *A_11, *A_12, *A_21, *A_22;
    getSubmatrixPointer(A, mDiv2, kDiv2, ldA, A_11, A_12, A_21, A_22);

    float *B_11, *B_12, *B_21, *B_22;
    getSubmatrixPointer(B, kDiv2, nDiv2, ldB, B_11, B_12, B_21, B_22);

    float *M_1A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_1B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M1 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_2A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_2B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M2 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_3A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_3B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M3 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_4A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_4B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M4 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_5A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_5B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M5 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_6A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_6B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M6 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    float *M_7A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    float *M_7B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    float *M7 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);

    cublasHandle_t *handleArray = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * 7);
    for (int i = 0; i < 7; ++i) {
        cublasCreate(handleArray + i);
    }

    cudaStream_t *streamArray = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 7);
    for (int i = 0; i < 7; ++i) {
        cudaStreamCreate(streamArray + i);
    }

    int id;
#pragma omp parallel private(id)
    {
#pragma omp sections
        {
#pragma omp section
            {
                // printf("thread ID: %d\n", omp_get_thread_num());
                matrixAdd(M_1A, A_11, A_22, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixAdd(M_1B, B_11, B_22, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handleArray[0], streamArray[0]);
                //gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handleArray[0]);
            }

#pragma omp section
            {
                // printf("thread ID: %d\n", omp_get_thread_num());
                matrixAdd(M_2A, A_21, A_22, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixCopy(M_2B, B_11, kDiv2, nDiv2, kDiv2, ldB);
                gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handleArray[1], streamArray[1]);
                //gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handleArray[1]);
            }
#pragma omp section
            {
                // printf("thread ID: %d\n", omp_get_thread_num());
                matrixCopy(M_3A, A_11, mDiv2, kDiv2, mDiv2, ldA);
                matrixMinus(M_3B, B_12, B_22, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handleArray[2], streamArray[2]);
                //gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handleArray[2]);
            }
#pragma omp section
            {
                // printf("thread ID: %d\n", omp_get_thread_num());
                matrixCopy(M_4A, A_22, mDiv2, kDiv2, mDiv2, ldA);
                matrixMinus(M_4B, B_21, B_11, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handleArray[3], streamArray[3]);
                //gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handleArray[3]);
            }
#pragma omp section
            {
                // printf("thread ID: %d\n", omp_get_thread_num());
                matrixAdd(M_5A, A_11, A_12, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixCopy(M_5B, B_22, kDiv2, nDiv2, kDiv2, ldB);
                gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handleArray[4], streamArray[4]);
                //gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handleArray[4]);
            }
#pragma omp section
            {
                // printf("thread ID: %d\n", omp_get_thread_num());
                matrixMinus(M_6A, A_21, A_11, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixAdd(M_6B, B_11, B_12, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handleArray[5], streamArray[5]);
                //gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handleArray[5]);
            }
#pragma omp section
            {
                // printf("thread ID: %d\n", omp_get_thread_num());
                matrixMinus(M_7A, A_12, A_22, mDiv2, kDiv2, mDiv2, ldA, ldA);
                matrixAdd(M_7B, B_21, B_22, kDiv2, nDiv2, kDiv2, ldB, ldB);
                gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handleArray[6], streamArray[6]);
                //gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handleArray[6]);
            }
        }
    }

    float *C_11, *C_12, *C_21, *C_22;
    getSubmatrixPointer(C, mDiv2, nDiv2, ldC, C_11, C_12, C_21, C_22);

#pragma omp parallel
    {
#pragma omp sections
        {
#pragma omp section
            {
                matrixAdd(C_11, C_11, M1, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_11, C_11, M4, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_11, M1, M4, mDiv2, nDiv2, ldC, mDiv2, mDiv2);
                //matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, ldC, ldC, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_21, C_21, M2, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_21, C_21, M4, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_21, M2, M4, mDiv2, nDiv2, ldC, mDiv2, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_12, C_12, M3, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_12, C_12, M5, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_12, M3, M5, mDiv2, nDiv2, ldC, mDiv2, mDiv2);
            }
#pragma omp section
            {
                matrixAdd(C_22, C_22, M1, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixMinus(C_22, C_22, M2, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, ldC, ldC, mDiv2);
                matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixMinus(C_22, M1, M2, mDiv2, nDiv2, ldC, mDiv2, mDiv2);
                //matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, ldC, ldC, mDiv2);
                //matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, ldC, ldC, mDiv2);
            }
        }
    }

    free(M_1A);
    free(M_1B);
    free(M1);
    free(M_2A);
    free(M_2B);
    free(M2);
    free(M_3A);
    free(M_3B);
    free(M3);
    free(M_4A);
    free(M_4B);
    free(M4);
    free(M_5A);
    free(M_5B);
    free(M5);
    free(M_6A);
    free(M_6B);
    free(M6);
    free(M_7A);
    free(M_7B);
    free(M7);

    for (int i = 0; i < 7; ++i) {
        cublasDestroy(handleArray[i]);
        cudaStreamDestroy(streamArray[i]);
    }
    free(handleArray);
    free(streamArray);
}

// gemm by strassen algorithm. compute C = AB.
// C, A and B must be consecutive.
void gemmstrassenNOomp(float *C, float *A, float *B, int m, int k, int n, cublasHandle_t handle) {
    int mDiv2 = m / 2;
    int kDiv2 = k / 2;
    int nDiv2 = n / 2;

    float *A_11, *A_12, *A_21, *A_22;
    getSubmatrixPointer(A, mDiv2, kDiv2, m, A_11, A_12, A_21, A_22);

    float *B_11, *B_12, *B_21, *B_22;
    getSubmatrixPointer(B, kDiv2, nDiv2, k, B_11, B_12, B_21, B_22);

    float *M_1A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixAdd(M_1A, A_11, A_22, mDiv2, kDiv2, mDiv2, m, m);
    float *M_1B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixAdd(M_1B, B_11, B_22, kDiv2, nDiv2, kDiv2, k, k);
    float *M1 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M1, M_1A, M_1B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_2A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixAdd(M_2A, A_21, A_22, mDiv2, kDiv2, mDiv2, m, m);
    float *M_2B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixCopy(M_2B, B_11, kDiv2, nDiv2, kDiv2, k);
    float *M2 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M2, M_2A, M_2B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_3A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixCopy(M_3A, A_11, mDiv2, kDiv2, mDiv2, m);
    float *M_3B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixMinus(M_3B, B_12, B_22, kDiv2, nDiv2, kDiv2, k, k);
    float *M3 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M3, M_3A, M_3B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_4A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixCopy(M_4A, A_22, mDiv2, kDiv2, mDiv2, m);
    float *M_4B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixMinus(M_4B, B_21, B_11, kDiv2, nDiv2, kDiv2, k, k);
    float *M4 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M4, M_4A, M_4B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_5A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixAdd(M_5A, A_11, A_12, mDiv2, kDiv2, mDiv2, m, m);
    float *M_5B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixCopy(M_5B, B_22, kDiv2, nDiv2, kDiv2, k);
    float *M5 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M5, M_5A, M_5B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_6A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixMinus(M_6A, A_21, A_11, mDiv2, kDiv2, mDiv2, m, m);
    float *M_6B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixAdd(M_6B, B_11, B_12, kDiv2, nDiv2, kDiv2, k, k);
    float *M6 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M6, M_6A, M_6B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *M_7A = (float*)malloc(sizeof(float) * mDiv2 * kDiv2);
    matrixMinus(M_7A, A_12, A_22, mDiv2, kDiv2, mDiv2, m, m);
    float *M_7B = (float*)malloc(sizeof(float) * kDiv2 * nDiv2);
    matrixAdd(M_7B, B_21, B_22, kDiv2, nDiv2, kDiv2, k, k);
    float *M7 = (float*)malloc(sizeof(float) * mDiv2 * nDiv2);
    gemmcublas(M7, M_7A, M_7B, mDiv2, kDiv2, nDiv2, handle, 0);

    float *C_11, *C_12, *C_21, *C_22;
    getSubmatrixPointer(C, mDiv2, nDiv2, m, C_11, C_12, C_21, C_22);

    matrixAdd(C_11, M1, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);
    matrixMinus(C_11, C_11, M5, mDiv2, nDiv2, m, m, mDiv2);
    matrixAdd(C_11, C_11, M7, mDiv2, nDiv2, m, m, mDiv2);

    matrixAdd(C_21, M2, M4, mDiv2, nDiv2, m, mDiv2, mDiv2);

    matrixAdd(C_12, M3, M5, mDiv2, nDiv2, m, mDiv2, mDiv2);

    matrixMinus(C_22, M1, M2, mDiv2, nDiv2, m, mDiv2, mDiv2);
    matrixAdd(C_22, C_22, M3, mDiv2, nDiv2, m, m, mDiv2);
    matrixAdd(C_22, C_22, M6, mDiv2, nDiv2, m, m, mDiv2);

    free(M_1A);
    free(M_1B);
    free(M1);
    free(M_2A);
    free(M_2B);
    free(M2);
    free(M_3A);
    free(M_3B);
    free(M3);
    free(M_4A);
    free(M_4B);
    free(M4);
    free(M_5A);
    free(M_5B);
    free(M5);
    free(M_6A);
    free(M_6B);
    free(M6);
    free(M_7A);
    free(M_7B);
    free(M7);
}
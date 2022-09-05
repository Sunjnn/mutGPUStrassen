#include <stdio.h>

#include "matrixUti.hxx"


void initMatrix(float *A, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        // A[i] = i;
        A[i] = 1;
    }
}

void gemm(float *C, float *A, float *B, int M, int K, int N) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            C[MAT_IDX(i, j, M)] = 0.0f;
            for (int k = 0; k < K; ++k) {
                C[MAT_IDX(i, j, M)] += A[MAT_IDX(i, k, M)] * B[MAT_IDX(k, j, K)];
            }
        }
    }
}

void getSubmatrixPointer(float *A, int mDiv2, int nDiv2, int ld, float *&A_11, float *&A_12, float *&A_21, float *&A_22) {
    A_11 = A;
    A_21 = A + mDiv2;
    A_12 = A + nDiv2 * ld;
    A_22 = A_12 + mDiv2;
}

void matrixAdd(float *C, float *A, float *B, int M, int N, int ldC, int ldA, int ldB) {
    // for (int j = 0; j < N; ++j) {
    //     for (int i = 0; i < M; ++i) {
    //         // C[i + j * ldC] = A[i + j * ldA] + B[i + j * ldB];
    //         C[MAT_IDX(i, j, ldC)] = A[MAT_IDX(i, j, ldA)] + B[MAT_IDX(i, j, ldB)];
    //     }
    // }

    for (int j = 0; j < N; ++j) {
        int MDiv8 = M / 8;
        for (int i = MDiv8 * 8; i < M; ++i) {
            C[MAT_IDX(i, j, ldC)] = A[MAT_IDX(i, j, ldA)] + B[MAT_IDX(i, j, ldB)];
        }
        for (int i = 0; i < MDiv8; ++i) {
            C[MAT_IDX(i * 8, j, ldC)] = A[MAT_IDX(i * 8, j, ldA)] + B[MAT_IDX(i * 8, j, ldB)];
            C[MAT_IDX(i * 8 + 1, j, ldC)] = A[MAT_IDX(i * 8 + 1, j, ldA)] + B[MAT_IDX(i * 8 + 1, j, ldB)];
            C[MAT_IDX(i * 8 + 2, j, ldC)] = A[MAT_IDX(i * 8 + 2, j, ldA)] + B[MAT_IDX(i * 8 + 2, j, ldB)];
            C[MAT_IDX(i * 8 + 3, j, ldC)] = A[MAT_IDX(i * 8 + 3, j, ldA)] + B[MAT_IDX(i * 8 + 3, j, ldB)];
            C[MAT_IDX(i * 8 + 4, j, ldC)] = A[MAT_IDX(i * 8 + 4, j, ldA)] + B[MAT_IDX(i * 8 + 4, j, ldB)];
            C[MAT_IDX(i * 8 + 5, j, ldC)] = A[MAT_IDX(i * 8 + 5, j, ldA)] + B[MAT_IDX(i * 8 + 5, j, ldB)];
            C[MAT_IDX(i * 8 + 6, j, ldC)] = A[MAT_IDX(i * 8 + 6, j, ldA)] + B[MAT_IDX(i * 8 + 6, j, ldB)];
            C[MAT_IDX(i * 8 + 7, j, ldC)] = A[MAT_IDX(i * 8 + 7, j, ldA)] + B[MAT_IDX(i * 8 + 7, j, ldB)];
        }
    }
}

void matrixMinus(float *C, float *A, float *B, int M, int N, int ldC, int ldA, int ldB) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            // C[i + j * ldC] = A[i + j * ldA] - B[i + j * ldB];
            C[MAT_IDX(i, j, ldC)] = A[MAT_IDX(i, j, ldA)] - B[MAT_IDX(i, j, ldB)];
        }
    }
}

void matrixCopy(float *C, float *A, int M, int N, int ldC, int ldA) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            // C[i + j * ldC] = A[i + j * ldA];
            C[MAT_IDX(i, j, ldC)] = A[MAT_IDX(i, j, ldA)];
        }
    }
}

void test(float *A, float *B, int M, int N) {
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float diff = A[MAT_IDX(i, j, M)] - B[MAT_IDX(i, j, M)];
            diff /= A[MAT_IDX(i, j, M)];
            if (diff > 0.001 || diff < -0.001) {
                printf("wrong result. i:%d j:%d %f %f\n", i, j, A[MAT_IDX(i, j, M)], B[MAT_IDX(i, j, M)]);
            }
        }
    }
}

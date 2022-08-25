#ifndef __MAT_IDX__
#define __MAT_IDX__
#define MAT_IDX(i, j, ld) (i + j * ld)
#endif

void initMatrix(float *A, int, int);

void gemm(float *C, float *A, float *B, int, int, int);

void getSubmatrixPointer(float *A, int mDiv2, int nDiv2, int ld, float *&A_11, float *&A_12, float *&A_21, float *&A_22);

void matrixAdd(float *C, float *A, float *B, int, int, int ldC, int ldA, int ldB);

void matrixMinus(float *C, float *A, float *B, int, int, int ldC, int ldA, int ldB);

void matrixCopy(float *C, float *A, int, int, int ldC, int ldA);

void test(float *A, float *B, int, int);

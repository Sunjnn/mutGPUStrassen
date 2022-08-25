#include <stdlib.h>
#include <cmath>

#include "blockMatrix.hxx"

blockMatrix::blockMatrix() {}

blockMatrix::blockMatrix(float *A, int m, int n, int ldA, int blockM, int blockN) {
    dimM = ceil(m / blockM);
    dimN = ceil(n / blockN);

    pointers = (float**)malloc(sizeof(float*) * dimM * dimN);
    for (int i = 0; i < dimM; ++i) {
        for (int j = 0; j < dimN; ++j) {
            pointers[i + j * dimM] = A + i * blockM + j * blockN * ldA;
        }
    }
}

float *blockMatrix::getBlockMatrix(int blockI, int blockJ) {
    return pointers[blockI + blockJ * dimM];
}

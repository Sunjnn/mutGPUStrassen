// seperate a matrix.
class blockMatrix {
public:
    float **pointers = nullptr;
    int dimM = 0;
    int dimN = 0;
    blockMatrix();
    blockMatrix(float *A, int m, int n, int ldA, int blockM, int blockN);
    float *getBlockMatrix(int blockI, int blockJ);
};

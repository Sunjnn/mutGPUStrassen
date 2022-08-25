
void gemmstrassen(float *C, float *A, float *B, int m, int k, int n);

void gemmstrassen(float *C, float *A, float *B, int m, int k, int n, int ldA, int ldB, int ldC);

void gemmstrassenNOomp(float *C, float *A, float *B, int m, int k, int n, cublasHandle_t handle);
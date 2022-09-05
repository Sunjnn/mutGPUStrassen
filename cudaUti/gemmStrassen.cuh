
void gemmstrassen(float *C, float *A, float *B, int m, int k, int n);

void gemmstrassen(float *C, float *A, float *B, int m, int k, int n, int ldA, int ldB, int ldC);

void gemmstrassen_v2(float *, int, float *, int, float *, int, int, int, int);

void gemmstrassenNOomp(float *C, float *A, float *B, int m, int k, int n, cublasHandle_t handle);

// void gemmstrassen_v3(float *, int, float *, int, float *, int, int);
// void gemmstrassen_v3(float*, int, float*, int, float*, int, int, cudaStream_t*, cublasHandle_t*);
// void gemmstrassen_v3(float*, int, float*, int, float*, int, int, cudaStream_t*, cublasHandle_t*, float*, float*);
// void gemmstrassen_v3(float*, int, float*, int, float*, int, int, cudaStream_t*, cublasHandle_t*, float*, float*, float*, float*, float*);
void gemmstrassen_v3(float*, int, float*, int, float*, int, int, cudaStream_t, cublasHandle_t, float*, float*, float*, float*, float*);


#ifndef __CHECKCUDA__
#define __CHECKCUDA__
#define CHECKCUDA(call)                                                     \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d. ", __FILE__, __LINE__);                       \
        printf("code:%d, reason:%s\n", error, cudaGetErrorString(error));   \
        exit(0);                                                            \
    }                                                                       \
}
#endif


#ifndef __CHECKCUBLAS__
#define __CHECKCUBLAS__
#define CHECKCUBLAS(call) {                                                                                             \
    const cublasStatus_t error = call;                                                                                  \
    if (error != CUBLAS_STATUS_SUCCESS) {                                                                               \
        printf("Error: %s:%d. ", __FILE__, __LINE__);                                                                   \
        printf("code: %d, name: %s, string: %s\n", error, cublasGetStatusName(error), cublasGetStatusString(error));    \
        exit(0);                                                                                                        \
    }                                                                                                                   \
}
#endif


void gemmcublas(float *C, float *A, float *B, int, int, int, cublasHandle_t handle, cudaStream_t stream);

void gemmcublas(float *C, float *A, float *B, int, int, int, cublasHandle_t handle);

int getdevicecount();

// float *getdeviceprop(int deviceCount);

float getdeviceprop(int dev);

__global__ void GPU8_mul(float *, int, float *, int, float *, int);

__global__ void GPU8_add(float *, int, float *, int, float *, int, int, int);

__global__ void GPU8_sub(float *, int, float *, int, float *, int, int, int);
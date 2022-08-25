#include "blockMatrix.hxx"


#ifndef THREADPOOL
#define THREADPOOL
class threadPool {
public:
    std::mutex mut;
    blockMatrix *bmatA;
    blockMatrix *bmatB;
    blockMatrix *bmatC;
    int m;
    int k;
    int n;
    int blockM;
    int blockK;
    int blockN;
    std::deque<int*> tasks;
    int deviceCount;
    float* memMiBs;

    threadPool() {}
    threadPool(float* C, float* A, float* B, int m, int k, int n, int blockM, int blockK, int blockN);
    void threadGPU(int dev);
    void threadCPU();
};
#endif

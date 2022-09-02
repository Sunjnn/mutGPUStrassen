#include <vector>
#include <deque>
#include <mutex>

#include "blockMatrix.hxx"


#ifndef THREADPOOL
#define THREADPOOL
class threadPoolConfig {
public:
    // std::mutex mut;
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
    // float* memMiBs;
    std::vector<int> GPUs;
    std::vector<float> memMiBs;

    threadPoolConfig();
    threadPoolConfig(float*, float*, float*, int, int, int, int, int, int, std::vector<int>);
};
void threadCPU(threadPoolConfig *config);
void threadGPUMas(threadPoolConfig *config, int dev, int threadNum, int start, int stop);
void threadGPUSub(threadPoolConfig *config, int start, int stop);
#endif

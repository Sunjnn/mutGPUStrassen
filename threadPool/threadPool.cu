#include <cublas_v2.h>

#include <vector>
#include <thread>
#include <functional>

#include "gemmStrassen.cuh"
#include "threadPool.cuh"
#include "cudaUti.cuh"


threadPoolConfig::threadPoolConfig(float* C, float* A, float* B, int M, int K, int N, int blockM, int blockK, int blockN): m(M), k(K), n(N), blockM(blockM), blockK(blockK), blockN(blockN) {
    bmatA = new blockMatrix(A, M, K, M, blockM, blockK);
    bmatB = new blockMatrix(B, K, N, K, blockK, blockN);
    bmatC = new blockMatrix(C, M, N, M, blockM, blockN);

    deviceCount = getdevicecount();
    memMiBs = getdeviceprop(deviceCount);
    for (int i = 0; i < bmatC->dimM; ++i) {
        for (int j = 0; j < bmatC->dimN; ++j) {
            //int task[2] = { i, j };
            int *task = new int[2];
            task[0] = i;
            task[1] = j;
            tasks.push_back(task);
        }
    }
}

void threadGPUSub(threadPoolConfig *config, int start, int stop) {
    for (int index = start; index < stop; ++index) {
        int i = config->tasks[index][0];
        int j = config->tasks[index][1];
        for (int k = 0; k < config->bmatA->dimN; ++k) {
            gemmstrassen(config->bmatC->getBlockMatrix(i, j), config->bmatA->getBlockMatrix(i, k), config->bmatB->getBlockMatrix(k, j), config->blockM, config->blockK, config->blockN, config->m, config->k, config->m);
        }
    }
}

void threadGPUMas(threadPoolConfig *config, int dev, int threadNum, int start, int stop) {
    CHECKCUDA(cudaSetDevice(dev));

    int iterLen = stop - start;
    int iterNumThread = iterLen / threadNum;
    std::thread **threadArray = (std::thread**)malloc(sizeof(std::thread*) * threadNum);
    // std::vector<std::thread> threadArray;

    int i = 0;
    for (i = 0; i < threadNum - 1; ++i) {
        // threadArray.push_back(std::move(std::thread(threadGPUSub, C, A, B, iterBegin + i * iterNumThread, iterBegin + (i + 1) * iterNumThread)));
        threadArray[i] = new std::thread(threadGPUSub, config, start + i * iterNumThread, start + (i + 1) * iterNumThread);
        
    }
    // threadArray.push_back(std::move(std::thread(threadGPUSub, C, A, B, iterBegin + i * iterNumThread, iterEnd)));
    threadArray[i] = new std::thread(threadGPUSub, config, start + i * iterNumThread, stop);
    

    for (i = 0; i < threadNum; ++i) {
        threadArray[i]->join();
    }
}

void threadCPU(threadPoolConfig *config) {
    float mem = (config->blockM * config->blockK + config->blockK * config->blockN + config->blockM * config->blockN) / 1048576.0f * 7.0f;
    int taskLenGPU = (config->tasks.end() - config->tasks.begin()) / config->deviceCount;
    std::thread **threadArray = (std::thread**)malloc(sizeof(std::thread*) * config->deviceCount);
    // std::vector<std::thread> threadArray;

    int dev = 0;
    int num_thread;
    for (dev = 0; dev < config->deviceCount - 1; ++dev)
    {
        num_thread = config->memMiBs[dev] / mem;
        if (num_thread > 3) num_thread = 3;

        // threadArray.push_back(std::move(std::thread(threadGPUMas, bmatC, bmatA, bmatB, num_thread, tasks.begin() + dev * taskLenGPU, tasks.begin() + (dev + 1) * taskLenGPU)));
        threadArray[dev] = new std::thread(threadGPUMas, config, dev, num_thread, dev * taskLenGPU, (dev + 1) * taskLenGPU);
    }
    num_thread = config->memMiBs[dev] / mem;
    if (num_thread > 16) num_thread = 16;
    // threadArray.push_back(std::move(std::thread(threadGPUMas, bmatC, bmatA, bmatB, tasks.begin() + dev * taskLenGPU, tasks.end())));
    printf("ready for threadGPUMas %d\n", dev);
    threadArray[dev] = new std::thread(threadGPUMas, config, dev, num_thread, dev * taskLenGPU, config->tasks.size());

    for (dev = 0; dev < config->deviceCount; ++dev) {
        threadArray[dev]->join();
    }
}
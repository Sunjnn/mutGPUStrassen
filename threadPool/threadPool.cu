#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>
#include <thread>
#include <functional>
#include <stdio.h>

#include "gemmStrassen.cuh"
#include "threadPool.cuh"
#include "cudaUti.cuh"
#include "matrixUti.hxx"

threadPoolConfig::threadPoolConfig() {}

threadPoolConfig::threadPoolConfig(float* C, float* A, float* B, int M, int K, int N, int blockM, int blockK, int blockN, std::vector<int> GPUsArray): m(M), k(K), n(N), blockM(blockM), blockK(blockK), blockN(blockN) {
    bmatA = new blockMatrix(A, M, K, M, blockM, blockK);
    bmatB = new blockMatrix(B, K, N, K, blockK, blockN);
    bmatC = new blockMatrix(C, M, N, M, blockM, blockN);

    deviceCount = getdevicecount();
    for (int i = 0; i < GPUsArray.size(); ++i) {
        if (GPUsArray[i] >= deviceCount) {
            printf("%d GPU does not exist.\n", GPUsArray[i]);
        }

        memMiBs.push_back(getdeviceprop(GPUsArray[i]));
        GPUs.push_back(GPUsArray[i]);
    }
    deviceCount = GPUsArray.size();

    for (int j = 0; j < bmatC->dimN; ++j) {
        for (int i = 0; i < bmatC->dimM; ++i) {
            //int task[2] = { i, j };
            int *task = new int[2];
            task[0] = i;
            task[1] = j;
            tasks.push_back(task);
        }
    }
}

void threadGPUSub(threadPoolConfig *config, int start, int stop) {
    float *CTmp = (float*)malloc(sizeof(float) * config->blockM * config->blockN);
    cudaStream_t *streamArray = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 3);
    cublasHandle_t *handleArray = (cublasHandle_t*)malloc(sizeof(cublasHandle_t) * 3);
    for (int i = 0; i < 3; ++i) {
        CHECKCUDA(cudaStreamCreate(streamArray + i));
        CHECKCUBLAS(cublasCreate(handleArray + i));
        CHECKCUBLAS(cublasSetStream(handleArray[i], streamArray[i]));
    }

    float *T1 = nullptr;
    CHECKCUDA(cudaMallocAsync(&T1, sizeof(float) * config->blockM * config->blockM, streamArray[0]));
    float *T2 = nullptr;
    CHECKCUDA(cudaMallocAsync(&T2, sizeof(float) * config->blockM * config->blockM, streamArray[1]));

    float *d_A = nullptr;
    CHECKCUDA(cudaMallocAsync(&d_A, sizeof(float) * config->blockM * config->blockM, streamArray[0]))

    float *d_B = nullptr;
    CHECKCUDA(cudaMallocAsync(&d_B, sizeof(float) * config->blockM * config->blockM, streamArray[1]));

    float *d_C = nullptr;
    CHECKCUDA(cudaMallocAsync(&d_C, sizeof(float) * config->blockM * config->blockM, streamArray[2]));

    for (int k = 0; k < config->bmatA->dimN; ++k) {
        for (int idx = start; idx < stop; ++idx) {
            int i = config->tasks[idx][0];
            int j = config->tasks[idx][1];
            CHECKCUBLAS(cublasSetMatrixAsync(config->blockM, config->blockM, sizeof(float), config->bmatA->getBlockMatrix(i, k), config->m, d_A, config->blockM, streamArray[0]));
            if (idx == start || config->tasks[idx][1] != config->tasks[idx - 1][1]) {
                // printf("HTD: i=%d, j=%d, k=%d\n", i, j, k);
                CHECKCUBLAS(cublasSetMatrixAsync(config->blockM, config->blockM, sizeof(float), config->bmatB->getBlockMatrix(k, j), config->k, d_B, config->blockM, streamArray[1]));
            }
            gemmstrassen_v3(CTmp, config->blockM, config->bmatA->getBlockMatrix(i, k), config->m, config->bmatB->getBlockMatrix(k, j), config->k, config->blockM, streamArray, handleArray, T1, T2, d_A, d_B, d_C);
            matrixAdd(config->bmatC->getBlockMatrix(i, j), config->bmatC->getBlockMatrix(i, j), CTmp, config->blockM, config->blockN, config->m, config->m, config->blockM);
        }
    }

    // for (int index = start; index < stop; ++index) {
    //     int i = config->tasks[index][0];
    //     int j = config->tasks[index][1];
    //     for (int k = 0; k < config->bmatA->dimN; ++k) {
    //         // gemmstrassen(config->bmatC->getBlockMatrix(i, j), config->bmatA->getBlockMatrix(i, k), config->bmatB->getBlockMatrix(k, j), config->blockM, config->blockK, config->blockN, config->m, config->k, config->m);
    //         // gemmstrassen_v2(config->bmatC->getBlockMatrix(i, j), config->m, config->bmatA->getBlockMatrix(i, k), config->m, config->bmatB->getBlockMatrix(k, j), config->k, config->blockM, config->blockK, config->blockN);

    //         gemmstrassen_v3(CTmp, config->blockM, config->bmatA->getBlockMatrix(i, k), config->m, config->bmatB->getBlockMatrix(k, j), config->k, config->blockM, streamArray, handleArray, T1, T2, d_A, d_B, d_C);
    //         matrixAdd(config->bmatC->getBlockMatrix(i, j), config->bmatC->getBlockMatrix(i, j), CTmp, config->blockM, config->blockN, config->m, config->m, config->blockM);
    //     }
    // }
    CHECKCUDA(cudaFreeAsync(T1, streamArray[0]));
    CHECKCUDA(cudaFreeAsync(T2, streamArray[1]));
    CHECKCUDA(cudaFreeAsync(d_A, streamArray[0]));
    CHECKCUDA(cudaFreeAsync(d_B, streamArray[1]));
    CHECKCUDA(cudaFreeAsync(d_C, streamArray[2]));
    free(CTmp);
    for (int i = 0; i < 3; ++i) {
        CHECKCUDA(cudaStreamDestroy(streamArray[i]));
        CHECKCUBLAS(cublasDestroy(handleArray[i]));
    }
    free(streamArray);
    free(handleArray);
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
    int maxThread = 4;
    for (dev = 0; dev < config->deviceCount - 1; ++dev)
    {
        num_thread = config->memMiBs[dev] / mem;
        if (num_thread > maxThread) num_thread = maxThread;

        // threadArray.push_back(std::move(std::thread(threadGPUMas, bmatC, bmatA, bmatB, num_thread, tasks.begin() + dev * taskLenGPU, tasks.begin() + (dev + 1) * taskLenGPU)));
        threadArray[dev] = new std::thread(threadGPUMas, config, config->GPUs[dev], num_thread, dev * taskLenGPU, (dev + 1) * taskLenGPU);
    }
    num_thread = config->memMiBs[dev] / mem;
    if (num_thread > maxThread) num_thread = maxThread;
    // threadArray.push_back(std::move(std::thread(threadGPUMas, bmatC, bmatA, bmatB, tasks.begin() + dev * taskLenGPU, tasks.end())));
    // printf("ready for threadGPUMas %d\n", dev);
    threadArray[dev] = new std::thread(threadGPUMas, config, config->GPUs[dev], num_thread, dev * taskLenGPU, config->tasks.size());

    for (dev = 0; dev < config->deviceCount; ++dev) {
        threadArray[dev]->join();
    }
}
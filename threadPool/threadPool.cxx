#include "threadPool.hxx"
#include "main.cuh"


threadPool::threadPool(float* C, float* A, float* B, int M, int K, int N, int blockM, int blockK, int blockN): m(m), k(k), n(n), blockM(blockM), blockK(blockK), blockN(blockN) {
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

void threadPool::threadGPU(int dev) {
    CHECKCUDA(cudaSetDevice(dev));
    while (1) {
        if (tasks.empty()) {
            break;
        }
        mut.lock();
        int *task = tasks.front();
        tasks.pop_front();
        mut.unlock();

        for (int k = 0; k < bmatA->dimN; ++k) {
            gemmstrassen(bmatC->getBlockMatrix(task[0], task[1]), bmatA->getBlockMatrix(task[0], k), bmatB->getBlockMatrix(k, task[1]), blockM, blockK, blockN, m, k, m);
        }
    }
}

void threadPool::threadCPU() {
    float mem = (blockM * blockK + blockK * blockN + blockM * blockN) / 1048576.0f * 7.0f;
#pragma omp parallel for num_threads(deviceCount)
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        int num_thread = memMiBs[dev] / mem;
#pragma omp parrallel for num_threads(num_thread)
        for (int threadID = 0; threadID < num_thread; ++threadID) {
            threadGPU(dev);
        }
    }
}
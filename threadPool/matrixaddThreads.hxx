#include <vector>
#include <thread>
#include <semaphore.h>
#include <mutex>

#include "threadPool.cuh"

#ifndef __MATRIXADDTHREADS__
#define __MATRIXADDTHREADS__
class matrixaddThreads {
public:
    std::thread **threadArray;
    int threadNum;
    // sem_t *semArray;
    sem_t sem;
    sem_t *semArrayMain;
    std::mutex mut;
    int exitFlag = 0;
    // threadPoolConfig *config;
    float **CTmp;
    int CTmpNum;
    std::vector<float*> CTasks;
    std::vector<int> idxTasks;
    matrixaddThreads() {}
    matrixaddThreads(int, int, int, int);
    void run(int, int, int);
    int getCTmp();
    void destory();
    void addTask(float*, int);
};
#endif
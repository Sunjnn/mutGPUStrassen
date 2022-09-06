#include "matrixaddThreads.hxx"
#include "matrixUti.hxx"

matrixaddThreads::matrixaddThreads(int threadNum, int CTmpNum, int blockM, int M): CTmpNum(CTmpNum), threadNum(threadNum) {
    CTmp = (float**)malloc(sizeof(float*) * CTmpNum);
    // semArray = (sem_t*)malloc(sizeof(sem_t) * CTmpNum);
    sem_init(&sem, 0, 0);
    semArrayMain = (sem_t*)malloc(sizeof(sem_t) * CTmpNum);
    for (int i = 0; i < CTmpNum; ++i) {
        CTmp[i] = (float*)malloc(sizeof(float) * blockM * blockM);
        // sem_init(semArray + i, 0, 0);
        sem_init(semArrayMain + i, 0, 1);
    }

    threadArray = (std::thread**)malloc(sizeof(std::thread*) * threadNum);
    for (int i = 0; i < threadNum; ++i) {
        threadArray[i] = new std::thread(&matrixaddThreads::run, this, i, blockM, M);
    }
}

void matrixaddThreads::run(int threadId, int blockM, int M) {
    while (1) {
        sem_wait(&sem);
        if (exitFlag && CTasks.empty()) {
            return;
        }

        // sem_wait(semArray[threadId]);
        mut.lock();
        float *CThread = CTasks.back();
        CTasks.pop_back();
        int idxThread = idxTasks.back();
        idxTasks.pop_back();
        mut.unlock();

        matrixAdd(CThread, CThread, CTmp[idxThread], blockM, blockM, M, M, blockM);
        sem_post(semArrayMain + idxThread);
    }
}

int matrixaddThreads::getCTmp() {
    int val;
    while (1) {
        for (int i = 0; i < CTmpNum; ++i) {
            sem_getvalue(semArrayMain + i, &val);
            if (val == 1) {
                sem_wait(semArrayMain + i);
                return i;
            }
        }
        printf("cannot find a free CTmp\n");
    }
}

void matrixaddThreads::destory() {
    exitFlag = 1;
    for (int i = 0; i < CTmpNum; ++i) {
        sem_post(&sem);
    }
    for (int i = 0; i < threadNum; ++i) {
        threadArray[i]->join();
        delete threadArray[i];
    }
    free(threadArray);

    for (int i = 0; i < CTmpNum; ++i) {
        // sem_destroy(semArray + i);
        sem_destroy(&sem);
        sem_destroy(semArrayMain + i);
    }
    // free(semArray);
    free(semArrayMain);
    free(CTmp);
}

void matrixaddThreads::addTask(float *C, int idxCTmp) {
    mut.lock();
    CTasks.push_back(C);
    idxTasks.push_back(idxCTmp);
    mut.unlock();
    sem_post(&sem);
}
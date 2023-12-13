#pragma once

#include <iostream>
#include <numa.h>
#include <omp.h>
#include <thread>

#define omp_parallel _Pragma("omp parallel")
#define omp_parallel_for _Pragma("omp parallel for") for
#define omp_parallel_for_1 _Pragma("omp parallel for schedule (static,1)") for
#define omp_parallel_for_256 _Pragma("omp parallel for schedule (static,256)") for

// 无openMP时c++使用
// #define parallel_for for
// #define parallel_for_1 for
// #define parallel_for_256 for

/***********************************************************************
 *                              【CPU INFO】
 ***********************************************************************/
static uint64_t ThreadNum = omp_get_max_threads();
static uint64_t SocketNum = numa_num_configured_nodes();
static uint64_t ThreadPerSocket = ThreadNum / SocketNum;

inline uint64_t getThreadSocketId(uint64_t threadId) { return threadId / ThreadPerSocket; }

inline uint64_t getThreadSocketOffset(uint64_t threadId) { return threadId % ThreadPerSocket; }

static void reSetThreadNum(uint64_t threadNum)
{
    omp_set_num_threads(threadNum);
    ThreadNum = threadNum;
    ThreadPerSocket = ThreadNum / SocketNum;
}

/* ******************************************************************************************************
 *                              【线程绑定】
 * 1.函数threadBindToCore(coreId) 用于将std::thread申请的线程绑定到固定的core上
 * 2.OMP的线程我们采用环境变量的方式绑定：
 *   ./bashrc中：
 *    export GOMP_CPU_AFFINITY="2 3 4 5 6 7 14 15 16 17 18 19"
 *    注意：这里的个数要与OMP_NUM_THREADS=12设定的个数对应上
 *         OMP_PROC_BIND的优先级比GOMP_CPU_AFFINITY高，因此设置GOMP_CPU_AFFINITY要取消OMP_PROC_BIND
 * ******************************************************************************************************/
bool threadBindToCore(int coreId)
{
    int nThreads = std::thread::hardware_concurrency();
    cpu_set_t set;
    CPU_ZERO(&set);

    if (coreId < 0 || coreId > nThreads)
    {
        printf("[Fail]: Input coreId invalid, coreId between [0, %u]\n", nThreads);
        return false;
    }

    CPU_SET(coreId, &set);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &set) < 0)
    {
        printf("[Fail]: Unable to Set Affinity\n");
        return false;
    }

    return true;
}

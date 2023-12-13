#pragma once

#include <cstdint> // uint32_t

#define WORDSIZE 64
#define RATE_UTIL 100000000 // Displays the percentage to print
#define PAGESIZE 4096       // PageSize 4096

#define BLOCKSIZE 256
#define WARPSIZE 32
#define HALFWARP 16
#define WARP_SHIFT 5
const constexpr uint32_t WARP_NUM_PER_BLOCK = BLOCKSIZE / WARPSIZE;
#define DONATE_POOL_SIZE 32

#define SUB_VERTEXSET 64     // 线程的划分chunk
#define VERTEXSTEAL_CHUNK 64 // 窃取线程调度顶点的chunk
#define VERTEXWORK_CHUNK 64
#define EDGESTEAL_THRESHOLD 12800 // 启用edgeSteal的阈值 12800
#define EDGESTEAL_CHUNK 6400      // 线程调度边的chunk 6400
#define VertexValue_MAX 999999999 // 这里不能用MAX命名, 会导致CUB出错

// #define GB(size) size /(1024*1024*1024)  // bytes to GB

/* **********************************************************************
 *                              【ALIGN】
 * **********************************************************************/
#define CACHE_ALIGNED __attribute__((aligned(64)))

/* **********************************************************************
 *                              【WORD】
 * **********************************************************************/
#define WORD_OFFSET(word) ((word) >> 6) // word / 64
#define WORD_MOD(word) ((word)&0x3f)    // word % 64

// 不检查类型
#define cpj_max(a, b) ((a > b) ? a : b)
#define cpj_min(a, b) ((a > b) ? b : a)

#define NUMA_AWARE

// 存储
#define KB(x) (static_cast<size_t>(x) << 10)
#define MB(x) (static_cast<size_t>(x) << 20)
#define GB(x) (static_cast<size_t>(x) << 30)

#define BYTES_TO_GB(bytes) ((bytes) / (1024.0 * 1024 * 1024))
#define BYTES_TO_MB(bytes) ((bytes) / (1024.0 * 1024))
#define BYTES_TO_KB(bytes) ((bytes) / 1024.0)

// static_cast
#define SC(x) (static_cast<uint64_t>(x))
#define SCast(x) (static_cast<uint64_t>(x))
#define SCU64(x) (static_cast<uint64_t>(x))
#define SCD(x) (static_cast<double>(x))
#define SCI32(x) (static_cast<int32_t>(x))

// PageRank
#define Alpha 0.85
#define Tolerance 0.0001
#define MaxIte_PR 5000
#define CHECK_Tolerance 0.01

// 文件路径符号
#define SEPARATOR (std::filesystem::path::preferred_separator)

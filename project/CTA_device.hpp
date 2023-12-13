#pragma once

#include "Basic/basic_include.cuh"

namespace CTA {
__device__ __forceinline__ unsigned int LaneId()
{
    unsigned int ret;
    asm("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

struct Thread_data_type
{
    degree_type nbrSize;
    countl_type offset_start;
    vertex_data_type msg; // vertexValue
};

struct Shared_data_type
{
    count_type owner; //
    degree_type nbrSize;
    countl_type offset_start;
    vertex_data_type msg; // vertexValue
};

template <typename Work_type> __device__ __forceinline__ static void schedule(Thread_data_type &thread_data, Work_type work)
{
    const count_type TB_SIZE = BLOCKSIZE;
    const count_type WARP_SIZE = WARPSIZE;

    __shared__ Shared_data_type shared;

    //
    if (threadIdx.x == 0)
    {
        shared.owner = 1025;
    }

    __syncthreads();

    // 256: block
    do
    {

        if (thread_data.nbrSize >= TB_SIZE)
            shared.owner = threadIdx.x;
        __syncthreads();

        if (shared.owner >= 1025)
        {
            __syncthreads();
            break;
        }

        if (shared.owner == threadIdx.x)
        {
            shared.owner = 1025;
            shared.nbrSize = thread_data.nbrSize;
            shared.offset_start = thread_data.offset_start;
            shared.msg = thread_data.msg;

            thread_data.nbrSize = 0;
        }
        __syncthreads();

        countl_type offset_start = shared.offset_start;
        degree_type nbrSize = shared.nbrSize;
        vertex_data_type msg = shared.msg;

        for (int tid = threadIdx.x; tid < nbrSize; tid += TB_SIZE)
        {
            work(offset_start + tid, msg);
        }
        __syncthreads();
    }
    while (true);

    // 32-256 : warp
    const int lane_id = LaneId();
    int mask = __ballot_sync(0xffffffff, thread_data.nbrSize >= WARP_SIZE);
    while (mask != 0)
    {

        int leader = __ffs(mask) - 1;

        // int clear_mask = ~(int(1) << (leader));
        mask &= ~(int(1) << (leader));

        degree_type nbrSize = __shfl_sync(0xffffffff, thread_data.nbrSize, leader);
        degree_type offset_start = __shfl_sync(0xffffffff, thread_data.offset_start, leader);
        vertex_data_type msg = __shfl_sync(0xffffffff, thread_data.msg, leader);

        if (leader == lane_id)
            thread_data.nbrSize = 0;

        for (int lid = lane_id; lid < nbrSize; lid += WARP_SIZE)
        {
            work(offset_start + lid, msg);
        }
    }

    // 32：thread
    for (degree_type tid = 0; tid < thread_data.nbrSize; tid++)
    {
        work(thread_data.offset_start + tid, thread_data.msg);
    }

} // end of func [schedule]

} // namespace CTA
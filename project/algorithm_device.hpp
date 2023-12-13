#pragma once

#include "Basic/GPU/cuda_func.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Type/data_type.hpp"
#include "Basic/basic_include.cuh"
#include "CTA_device.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

typedef vertex_id_type emogi_type;

#define MEM_ALIGN_64 (~(0xfULL))
#define MEM_ALIGN_32 (~(0x1fULL))

template <typename T> struct MemAlignTrait
{
    static const emogi_type value = static_cast<emogi_type>(MEM_ALIGN_32); // 默认值为MEM_ALIGN_32，表示不支持的类型
    constexpr static const vertex_data_type vertexValue = std::numeric_limits<emogi_type>::max();
};

template <> struct MemAlignTrait<uint32_t>
{
    static const emogi_type value = static_cast<emogi_type>(MEM_ALIGN_32); // 32位时的对齐值
    constexpr static const vertex_data_type vertexValue = std::numeric_limits<emogi_type>::max();
};

template <> struct MemAlignTrait<uint64_t>
{
    static const emogi_type value = static_cast<emogi_type>(MEM_ALIGN_64); // 64位时的对齐值
    constexpr static const vertex_data_type vertexValue = std::numeric_limits<emogi_type>::max();
};

constexpr emogi_type MEM_ALIGN_CPJ = MemAlignTrait<emogi_type>::value;
constexpr vertex_data_type MYINFINITY_cpj = MemAlignTrait<vertex_data_type>::vertexValue;

#define MYINFINITY_CPJ 0XFFFFFFFF

#define CHUNK_SHIFT_EMOGI 3
#define CHUNK_SIZE_EMOGI (1 << CHUNK_SHIFT_EMOGI)

/* *********************************************************************************************************
 * @description: BFS
 * *********************************************************************************************************/
namespace BFS_DEVICE_SPACE {

namespace CTA_imp {
//> 2. [CTA_imp]: worklist_model_kernel
__global__ void worklist_model_kernel(countl_type *common, vertex_id_type *worklist_in, vertex_id_type *worklist_out, countl_type *csr_offset,
                                      vertex_id_type *csr_dest, vertex_data_type *vertexValue)
{
    count_type workNum_up = gridDim.x * blockDim.x;
    for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < workNum_up; threadId += workNum_up)
    {
        CTA::Thread_data_type thread_data = {0, 0, 0};

        if (threadId < common[1]) // worklist_size
        {
            vertex_id_type vertexId = worklist_in[threadId];
            thread_data.offset_start = csr_offset[vertexId];
            thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
            thread_data.msg = vertexValue[vertexId];
        }

        CTA::schedule(thread_data,
                      [&](countl_type edge_cur, vertex_data_type msg)
                      {
            vertex_id_type dest = csr_dest[edge_cur];
            if (msg + 1 < vertexValue[dest])
            {
                if (msg + 1 < cuda_atomicMin((vertexValue + dest), msg + 1))
                {
                    // countl_type index = 0; //$$$= atomicAdd(common, 1); // worklist_count //!-------------------------
                    countl_type index = 0;
                    index = cuda_atomicAdd(common, 1);
                    worklist_out[index] = dest;
                }
            }
        });
    }
}

//> 1. [CTA_imp]: worklist_model_donatewarp
template <typename GraphDevice> void worklist_model(GraphDevice &graphDevice, const count_type nBlock)
{
    worklist_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.common_device, graphDevice.worklist_device.in(), graphDevice.worklist_device.out(),
                                                 graphDevice.csr_offset_device, graphDevice.csr_dest_device, graphDevice.vertexValue_in_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace CTA_imp

//> 2. [Baseline]: baseline_model
__global__ void baseline_model_kernel(vertex_data_type *vertexValue, const int level, const count_type vertex_count, const countl_type *csr_offset,
                                      const vertex_id_type *csr_dest, countl_type *changed)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadId < vertex_count && vertexValue[threadId] == level)
    {
        const uint64_t start = csr_offset[threadId];
        const uint64_t end = csr_offset[threadId + 1];

        for (countl_type i = start; i < end; i++)
        {
            vertex_id_type next = csr_dest[i];

            if (vertexValue[next] == MYINFINITY_cpj) // 0xFFFFFFFF
            {
                vertexValue[next] = level + 1;
                changed[0] = true;
            }
        }
    }
}

//> 1. [Baseline]: baseline_model
template <typename GraphDevice> void baseline_model(GraphDevice &graphDevice, const count_type nBlock, int level)
{
    baseline_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.vertexValue_in_device, level, graphDevice.vertexNum, graphDevice.csr_offset_device,
                                                 graphDevice.csr_dest_device, graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2. [Coalesce]: coalesce_model
__global__ void coalesce_model_kernel(vertex_data_type *vertexValue, const int level, const count_type vertex_count, const countl_type *csr_offset,
                                      const vertex_id_type *csr_dest, countl_type *changed)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const count_type warpIdx = threadId >> WARP_SHIFT;
    const count_type laneIdx = threadId & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertex_count && vertexValue[warpIdx] == level)
    {
        const uint64_t start = csr_offset[warpIdx];
        // 32位对齐时, 0-32 =>0, 32-63 => 32, 64-95 => 64
        // 64为对齐时, 0-15 =>0, 16-32 => 16, 32-47 => 32
        const uint64_t shift_start = start & MEM_ALIGN_CPJ;
        const uint64_t end = csr_offset[warpIdx + 1];

        for (uint64_t i = shift_start + laneIdx; i < end; i += WARPSIZE)
        {
            if (i >= start)
            {
                const vertex_id_type next = csr_dest[i];

                if (vertexValue[next] == MYINFINITY_cpj)
                {
                    vertexValue[next] = level + 1;
                    changed[0] = true;
                }
            }
        }
    }
}

//> 1. [Coalesce]: coalesce_model
template <typename GraphDevice> void coalesce_model(GraphDevice &graphDevice, const count_type nBlock, int level)
{
    coalesce_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.vertexValue_in_device, level, graphDevice.vertexNum, graphDevice.csr_offset_device,
                                                 graphDevice.csr_dest_device, graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2. [coalesce_chunk] coalesce_chunk_model_kernel
__global__ void coalesce_chunk_model_kernel(vertex_data_type *vertexValue, const int level, const count_type vertex_count,
                                            const countl_type *csr_offset, const vertex_id_type *csr_dest, countl_type *changed)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertex_count)
    {
        if (vertex_count > chunkIdx)
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        if (vertexValue[i] == level)
        {
            const uint64_t start = csr_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            const uint64_t end = csr_offset[i + 1];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            {
                if (j >= start)
                {
                    const vertex_id_type next = csr_dest[j];

                    if (vertexValue[next] == MYINFINITY_cpj)
                    {
                        vertexValue[next] = level + 1;
                        *changed = true;
                    }
                }
            }
        }
    }
}

//> 1. [coalesce_chunk] coalesce_chunk_model
template <typename GraphDevice> void coalesce_chunk_model(GraphDevice &graphDevice, const count_type nBlock, int level)
{
    coalesce_chunk_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.vertexValue_in_device, level, graphDevice.vertexNum, graphDevice.csr_offset_device,
                                                       graphDevice.csr_dest_device, graphDevice.common_device);

    // //CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2. [coalesce_chunk] coalesce_chunk_coProcess_model_kernel
__global__ void coalesce_chunk_coProcess_model_kernel(vertex_data_type *vertexValue, const int level, const count_type vertex_count,
                                                      const countl_type *csr_offset, const vertex_id_type *csr_dest)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertex_count)
    {
        if (vertex_count > chunkIdx)
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        if (vertexValue[i] == level)
        {
            const uint64_t start = csr_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            const uint64_t end = csr_offset[i + 1];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            {
                if (j >= start)
                {
                    const vertex_id_type next = csr_dest[j];

                    if (vertexValue[next] == MYINFINITY_cpj)
                    {
                        vertexValue[next] = level + 1;
                    }
                }
            }
        }
    }
}

//> 1. [coalesce_chunk] coalesce_chunk_coProcess_model
template <typename GraphDevice>
void coalesce_chunk_coProcess_model(GraphDevice &graphDevice, const count_type nBlock, int level, count_type vertexNumProcess)
{
    coalesce_chunk_coProcess_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.vertexValue_in_device, level, vertexNumProcess,
                                                                 graphDevice.copResult.csrResult_device.csr_offset,
                                                                 graphDevice.copResult.csrResult_device.csr_dest);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2. [coalesce_chunk] coalesce_chunk_coProcess_model_kernel
__global__ void coalesce_chunk_coProcess_help_model_kernel(vertex_data_type *vertexValue, const int level, const count_type vertex_count,
                                                           const count_type startVertexId_GPU_ZEROCOPY, const countl_type *csr_offset,
                                                           const vertex_id_type *csr_dest)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertex_count)
    {
        if (vertex_count > chunkIdx)
            chunk_size = vertex_count - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        vertex_id_type curVertexId = i + startVertexId_GPU_ZEROCOPY;
        if (vertexValue[curVertexId] == level)
        {
            const uint64_t start = csr_offset[curVertexId];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            const uint64_t end = csr_offset[curVertexId + 1];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            {
                if (j >= start)
                {
                    const vertex_id_type next = csr_dest[j];

                    if (vertexValue[next] == MYINFINITY_cpj)
                    {
                        vertexValue[next] = level + 1;
                    }
                }
            }
        }
    }
}

//> 1. [coalesce_chunk] coalesce_chunk_coProcess_model
template <typename GraphDevice>
void coalesce_chunk_coProcess_help_model(GraphDevice &graphDevice, const count_type nBlock, int level, count_type vertexNumProcess,
                                         count_type startVertexId_GPU_ZEROCOPY)
{
    coalesce_chunk_coProcess_help_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.vertexValue_in_device, level, vertexNumProcess,
                                                                      startVertexId_GPU_ZEROCOPY, graphDevice.copResult.csrResult_host.csr_offset,
                                                                      graphDevice.copResult.csrResult_host.csr_dest);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace BFS_DEVICE_SPACE

//!------------------------------------------------------------------------------------------------------------------------------------------------------

/* *********************************************************************************************************
 * @description: SSSP
 * *********************************************************************************************************/
namespace SSSP_DEVICE_SPACE {

namespace CTA_imp {
//> 2. [CTA_imp]: worklist_model_kernel
__global__ void worklist_model_kernel(countl_type *common, vertex_id_type *worklist_in, vertex_id_type *worklist_out, countl_type *csr_offset,
                                      vertex_id_type *csr_dest, edge_data_type *csr_weight, vertex_data_type *vertexValue)
{
    count_type workNum_up = gridDim.x * blockDim.x;
    for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < workNum_up; threadId += workNum_up)
    {
        CTA::Thread_data_type thread_data = {0, 0, 0};

        if (threadId < common[1]) // worklist_size
        {
            vertex_id_type vertexId = worklist_in[threadId];
            thread_data.offset_start = csr_offset[vertexId];
            thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
            thread_data.msg = vertexValue[vertexId];
        }

        CTA::schedule(thread_data,
                      [&](countl_type edge_cur, vertex_data_type msg)
                      {
            vertex_id_type dest = csr_dest[edge_cur];
            edge_data_type weight = csr_weight[edge_cur];
            if (msg + weight < vertexValue[dest])
            {
                if (msg + weight < cuda_atomicMin((vertexValue + dest), msg + weight))
                {
                    // countl_type index = 0; //$$$= atomicAdd(common, 1); // worklist_count //!--------------------------
                    countl_type index = 0;
                    index = cuda_atomicAdd(common, 1);
                    worklist_out[index] = dest;
                    // GPU_info("index = %d, dest = %d", index, dest);
                }
            }
        });
    }
}

//> 1. [CTA_imp]: worklist_model_donatewarp
template <typename GraphDevice> void worklist_model(GraphDevice &graphDevice, const count_type nBlock)
{
    worklist_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.common_device, graphDevice.worklist_device.in(), graphDevice.worklist_device.out(),
                                                 graphDevice.csr_offset_device, graphDevice.csr_dest_device, graphDevice.csr_weight_device,
                                                 graphDevice.vertexValue_in_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace CTA_imp

//> 2. [Coalesce]: coalesce_model
__global__ void coalesce_model_kernel(bool *label, vertex_data_type *vertexValue_in, const count_type vertexNum, countl_type *csr_offset,
                                      vertex_id_type *csr_dest, edge_data_type *csr_weight)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const count_type warpIdx = threadId >> WARP_SHIFT;
    const count_type laneIdx = threadId & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertexNum && label[warpIdx])
    {
        uint64_t start = csr_offset[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN_CPJ;
        uint64_t end = csr_offset[warpIdx + 1];

        vertex_data_type cost = vertexValue_in[warpIdx];

        for (uint64_t i = shift_start + laneIdx; i < end; i += WARPSIZE)
        {
            if (vertexValue_in[warpIdx] != cost) // TODO
                break;
            if (vertexValue_in[csr_dest[i]] > cost + csr_weight[i] && i >= start)
                cuda_atomicMin(&(vertexValue_in[csr_dest[i]]), cost + csr_weight[i]);
        }

        label[warpIdx] = false;
    }
}

//> UPDATE
__global__ void update(bool *label, vertex_data_type *vertexValue_in, vertex_data_type *vertexValue_out, const count_type vertexNum,
                       countl_type *changed)
{
    // uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < vertexNum)
    {
        if (vertexValue_in[threadId] < vertexValue_out[threadId])
        {
            vertexValue_out[threadId] = vertexValue_in[threadId];
            label[threadId] = true;
            *changed = true;
        }
    }
}

//> 1. [Coalesce]: coalesce_model
template <typename GraphDevice> void coalesce_model(GraphDevice &graphDevice, const count_type nBlock, const count_type nBlock_update)
{
    coalesce_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_in_device, graphDevice.vertexNum,
                                                 graphDevice.csr_offset_device, graphDevice.csr_dest_device, graphDevice.csr_weight_device);

    update<<<nBlock_update, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_in_device, graphDevice.vertexValue_out_device,
                                         graphDevice.vertexNum, graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2. 2. [Coalesce_chunk]: coalesce_chunk_model_kernel
__global__ void coalesce_chunk_model_kernel(bool *label, vertex_data_type *vertexValue_in, const count_type vertexNum, countl_type *csr_offset,
                                            vertex_id_type *csr_dest, edge_data_type *csr_weight)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertexNum)
    {
        if (vertexNum > chunkIdx)
            chunk_size = vertexNum - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        if (label[i])
        {
            uint64_t start = csr_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            uint64_t end = csr_offset[i + 1];

            vertex_data_type cost = vertexValue_in[i];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            {
                if (vertexValue_in[i] != cost)
                    break;
                if (vertexValue_in[csr_dest[j]] > cost + csr_weight[j] && j >= start)
                    cuda_atomicMin(&(vertexValue_in[csr_dest[j]]), cost + csr_weight[j]);
            }

            label[i] = false;
        }
    }
}

//> 1. [Coalesce_chunk]: coalesce_chunk_model
template <typename GraphDevice> void coalesce_chunk_model(GraphDevice &graphDevice, const count_type nBlock, const count_type nBlock_update)
{
    coalesce_chunk_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_in_device, graphDevice.vertexNum,
                                                       graphDevice.csr_offset_device, graphDevice.csr_dest_device, graphDevice.csr_weight_device);

    update<<<nBlock_update, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_in_device, graphDevice.vertexValue_out_device,
                                         graphDevice.vertexNum, graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2.[Coalesce_chunk]: coalesce_chunk_coProcess_model_kernel
__global__ void coalesce_chunk_coProcess_model_kernel(size_t *label, vertex_data_type *vertexValue_in, const count_type vertexNum,
                                                      countl_type *csr_offset, vertex_id_type *csr_dest, edge_data_type *csr_weight)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertexNum)
    {
        if (vertexNum > chunkIdx)
            chunk_size = vertexNum - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        size_t arrpos = BIT_OFFSET(i);
        size_t bitpos = BIT_MOD(i);
        size_t one = (size_t(1) << size_t(bitpos));
        if (label[arrpos] & one)
        {
            uint64_t start = csr_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            uint64_t end = csr_offset[i + 1];

            vertex_data_type cost = vertexValue_in[i];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            {
                if (vertexValue_in[i] != cost)
                    break;
                if (vertexValue_in[csr_dest[j]] > cost + csr_weight[j] && j >= start)
                    cuda_atomicMin(&(vertexValue_in[csr_dest[j]]), cost + csr_weight[j]);
            }
        }
    }
}

//> 1. [Coalesce_chunk]: coalesce_chunk_coProcess_model
template <typename GraphDevice> void coalesce_chunk_coProcess_model(GraphDevice &graphDevice, const count_type nBlock, count_type vertexNumProcess)
{
    coalesce_chunk_coProcess_model_kernel<<<nBlock, BLOCKSIZE>>>(
        graphDevice.label_cur_device, graphDevice.vertexValue_in_device, vertexNumProcess, graphDevice.copResult.csrResult_device.csr_offset,
        graphDevice.copResult.csrResult_device.csr_dest, graphDevice.copResult.csrResult_device.csr_weight);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2.[Coalesce_chunk]: coalesce_chunk_coProcess_model_kernel
__global__ void coalesce_chunk_coProcess_help_model_kernel(size_t *label, vertex_data_type *vertexValue_in, const count_type vertexNum,
                                                           const count_type startVertexId_GPU_ZEROCOPY, countl_type *csr_offset,
                                                           vertex_id_type *csr_dest, edge_data_type *csr_weight)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertexNum)
    {
        if (vertexNum > chunkIdx)
            chunk_size = vertexNum - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        vertex_id_type curVertexId = i + startVertexId_GPU_ZEROCOPY;
        size_t arrpos = BIT_OFFSET(curVertexId);
        size_t bitpos = BIT_MOD(curVertexId);
        size_t one = (size_t(1) << size_t(bitpos));
        if (label[arrpos] & one)
        {
            uint64_t start = csr_offset[curVertexId];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            uint64_t end = csr_offset[curVertexId + 1];

            vertex_data_type cost = vertexValue_in[curVertexId];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            {
                if (vertexValue_in[curVertexId] != cost)
                    break;
                if (vertexValue_in[csr_dest[j]] > cost + csr_weight[j] && j >= start)
                    cuda_atomicMin(&(vertexValue_in[csr_dest[j]]), cost + csr_weight[j]);
            }
        }
    }
}

//> 1. [Coalesce_chunk]: coalesce_chunk_coProcess_model
template <typename GraphDevice>
void coalesce_chunk_coProcess_help_model(GraphDevice &graphDevice, const count_type nBlock, count_type vertexNumProcess,
                                         count_type startVertexId_GPU_ZEROCOPY)
{
    coalesce_chunk_coProcess_help_model_kernel<<<nBlock, BLOCKSIZE>>>(
        graphDevice.label_cur_device, graphDevice.vertexValue_in_device, vertexNumProcess, startVertexId_GPU_ZEROCOPY,
        graphDevice.copResult.csrResult_host.csr_offset, graphDevice.copResult.csrResult_host.csr_dest,
        graphDevice.copResult.csrResult_host.csr_weight);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace SSSP_DEVICE_SPACE

//!-------------------------------------------------------------------------------------------------------------------------------------------
namespace WCC_DEVICE_SPACE {

namespace CTA_imp {
//> 2. [CTA_imp]: worklist_model_kernel
__global__ void worklist_model_kernel(countl_type *common, vertex_id_type *worklist_in, vertex_id_type *worklist_out, countl_type *csr_offset,
                                      vertex_id_type *csr_dest, vertex_data_type *vertexValue)
{
    count_type workNum_up = gridDim.x * blockDim.x;
    for (count_type threadId = threadIdx.x + blockIdx.x * blockDim.x; threadId < workNum_up; threadId += workNum_up)
    {
        CTA::Thread_data_type thread_data = {0, 0, 0};

        if (threadId < common[1]) // worklist_size
        {
            vertex_id_type vertexId = worklist_in[threadId];
            thread_data.offset_start = csr_offset[vertexId];
            thread_data.nbrSize = csr_offset[vertexId + 1] - thread_data.offset_start;
            thread_data.msg = vertexValue[vertexId];
        }

        CTA::schedule(thread_data,
                      [&](countl_type edge_cur, vertex_data_type msg)
                      {
            vertex_id_type dest = csr_dest[edge_cur];
            if (msg < vertexValue[dest])
            {
                if (msg < cuda_atomicMin((vertexValue + dest), msg))
                {
                    // uint32_t index = 0; //$$$atomicAdd(common, 1); // worklist_count//!----------------------------
                    countl_type index = 0;
                    index = cuda_atomicAdd(common, 1);
                    worklist_out[index] = dest;
                }
            }
        });
    }
}

//> 1. [CTA_imp]: worklist_model_donatewarp
template <typename GraphDevice> void worklist_model(GraphDevice &graphDevice, const count_type nBlock)
{
    worklist_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.common_device, graphDevice.worklist_device.in(), graphDevice.worklist_device.out(),
                                                 graphDevice.csr_offset_device, graphDevice.csr_dest_device, graphDevice.vertexValue_in_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace CTA_imp

//> 2. [Coalesce]: coalesce_model
__global__ void coalesce_model_kernel(bool *curr_visit, bool *next_visit, const count_type vertexNum, countl_type *csr_offset,
                                      vertex_id_type *csr_dest, vertex_data_type *vertexValue_in, countl_type *common)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const count_type warpIdx = threadId >> WARP_SHIFT;
    const count_type laneIdx = threadId & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertexNum && curr_visit[warpIdx] == true)
    {
        const uint64_t start = csr_offset[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN_CPJ;
        const uint64_t end = csr_offset[warpIdx + 1];

        for (uint64_t i = shift_start + laneIdx; i < end; i += WARPSIZE)
        {
            if (i >= start)
            {
                vertex_data_type comp_src = vertexValue_in[warpIdx];
                vertex_id_type next = csr_dest[i];

                vertex_data_type comp_next = vertexValue_in[next];
                vertex_data_type comp_target;
                vertex_id_type next_target;

                if (comp_next != comp_src) // 同一边的src与dest的值不一样
                {
                    if (comp_src < comp_next)
                    {
                        next_target = next;
                        comp_target = comp_src;
                    }
                    else
                    {
                        next_target = warpIdx;
                        comp_target = comp_next;
                    }

                    cuda_atomicMin(&vertexValue_in[next_target], comp_target);
                    next_visit[next_target] = true;
                    *common = true;
                }
            }
        }
    }
}

//> 1. [Coalesce]: coalesce_model
template <typename GraphDevice> void coalesce_model(GraphDevice &graphDevice, const count_type nBlock)
{
    coalesce_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.label_device.in(), graphDevice.label_device.out(), graphDevice.vertexNum,
                                                 graphDevice.csr_offset_device, graphDevice.csr_dest_device, graphDevice.vertexValue_in_device,
                                                 graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2. [coalesce_chunk] coalesce_chunk_model_kernel
__global__ void coalesce_chunk_model_kernel(bool *curr_visit, bool *next_visit, const count_type vertexNum, countl_type *csr_offset,
                                            vertex_id_type *csr_dest, vertex_data_type *vertexValue_in, countl_type *common)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertexNum)
    {
        if (vertexNum > chunkIdx)
            chunk_size = vertexNum - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        if (curr_visit[i])
        {
            const uint64_t start = csr_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            const uint64_t end = csr_offset[i + 1];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            {
                if (j >= start)
                {
                    vertex_data_type comp_src = vertexValue_in[i];
                    vertex_id_type next = csr_dest[j];

                    vertex_data_type comp_next = vertexValue_in[next];
                    vertex_data_type comp_target;
                    vertex_id_type next_target;

                    if (comp_next != comp_src)
                    {
                        if (comp_src < comp_next)
                        {
                            next_target = next;
                            comp_target = comp_src;
                        }
                        else
                        {
                            next_target = i;
                            comp_target = comp_next;
                        }

                        cuda_atomicMin(&vertexValue_in[next_target], comp_target);
                        next_visit[next_target] = true;
                        *common = true;
                    }
                }
            }
        }
    }
}

//> 1. [coalesce_chunk] coalesce_chunk_model
template <typename GraphDevice> void coalesce_chunk_model(GraphDevice &graphDevice, const count_type nBlock)
{
    coalesce_chunk_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.label_device.in(), graphDevice.label_device.out(), graphDevice.vertexNum,
                                                       graphDevice.csr_offset_device, graphDevice.csr_dest_device, graphDevice.vertexValue_in_device,
                                                       graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 2.[Coalesce_chunk]: coalesce_chunk_coProcess_model_kernel
__global__ void coalesce_chunk_coProcess_model_kernel(size_t *curr_visit, vertex_data_type *vertexValue_in, const count_type vertexNum,
                                                      countl_type *csr_offset, vertex_id_type *csr_dest)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);
    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertexNum)
    {
        if (vertexNum > chunkIdx)
            chunk_size = vertexNum - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        size_t arrpos = BIT_OFFSET(i);
        size_t bitpos = BIT_MOD(i);
        size_t one = (size_t(1) << size_t(bitpos));
        if (curr_visit[arrpos] & one)
        {
            const uint64_t start = csr_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            const uint64_t end = csr_offset[i + 1];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            {
                if (j >= start)
                {
                    vertex_data_type comp_src = vertexValue_in[i];
                    vertex_id_type next = csr_dest[j];

                    vertex_data_type comp_next = vertexValue_in[next];
                    vertex_data_type comp_target;
                    vertex_id_type next_target;

                    if (comp_next != comp_src)
                    {
                        if (comp_src < comp_next)
                        {
                            next_target = next;
                            comp_target = comp_src;
                        }
                        else
                        {
                            next_target = i;
                            comp_target = comp_next;
                        }

                        cuda_atomicMin(&vertexValue_in[next_target], comp_target);
                    }
                }
            }
        }
    }
}

//> 1. [coalesce_chunk] coalesce_chunk_coProcess_model
template <typename GraphDevice> void coalesce_chunk_coProcess_model(GraphDevice &graphDevice, const count_type nBlock, count_type vertexNumProcess)
{
    coalesce_chunk_coProcess_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_in_device, vertexNumProcess,
                                                                 graphDevice.copResult.csrResult_device.csr_offset,
                                                                 graphDevice.copResult.csrResult_device.csr_dest);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace WCC_DEVICE_SPACE

namespace PR_DEVICE_SPACE {

__global__ void update_kernel(bool *label, vertex_data_type *vertexValue_in, vertex_data_type *vertexValue_out, const count_type vertexNum,
                              const countl_type *csr_offset, countl_type *common)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    // if (threadId < vertexNum)
    //     printf("tid[%d] = %f\n", threadId, vertexValue_in[threadId]);
    if (threadId < vertexNum)
    {
        vertexValue_out[threadId] = (1.0 - Alpha) / vertexNum + Alpha * vertexValue_out[threadId];
        if (csr_offset[threadId + 1] - csr_offset[threadId] > 0)
        {
            vertexValue_out[threadId] /= (csr_offset[threadId + 1] - csr_offset[threadId]);
        }
        if ((vertexValue_out[threadId] - vertexValue_in[threadId]) > Tolerance)
        {
            // printf("[Not Ok]: vertexId = %d, in = %f, out = %f\n", threadId, vertexValue_in[threadId], vertexValue_out[threadId]);
            vertexValue_in[threadId] = 0.0f;
            label[threadId] = true;
            *common = true;
        }
        else
        {
            vertexValue_in[threadId] = 0.0f;
        }
    }
}

//> 1. [Coalesce]: coalesce_model_kernel
__global__ void coalesce_model_kernel(bool *label, vertex_data_type *vertexValue_in, vertex_data_type *vertexValue_out, const count_type vertexNum,
                                      const countl_type *csr_offset, const vertex_id_type *csr_dest)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);

    if (warpIdx < vertexNum && label[warpIdx])
    {
        // if (threadId < 8)
        //     printf("[come]tid = %d, in =%f, out = %f\n", threadId, vertexValue_in[threadId], vertexValue_out[threadId]);
        const uint64_t start = csr_offset[warpIdx];
        const uint64_t shift_start = start & MEM_ALIGN_CPJ;
        const uint64_t end = csr_offset[warpIdx + 1];

        for (uint64_t i = shift_start + laneIdx; i < end; i += WARPSIZE)
            if (i >= start)
                atomicAdd(&vertexValue_out[csr_dest[i]], vertexValue_in[warpIdx]);

        label[warpIdx] = false;
    }
}

//> 1. [Coalesce]: coalesce_model
template <typename GraphDevice> void coalesce_model(GraphDevice &graphDevice, const count_type nBlock, const count_type nBlock_update)
{
    coalesce_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_device.in(),
                                                 graphDevice.vertexValue_device.out(), graphDevice.vertexNum, graphDevice.csr_offset_device,
                                                 graphDevice.csr_dest_device);

    update_kernel<<<nBlock_update, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_device.in(),
                                                graphDevice.vertexValue_device.out(), graphDevice.vertexNum, graphDevice.csr_offset_device,
                                                graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 1. [Coalesce]: coalesce_chunk_model_kernel
__global__ void coalesce_chunk_model_kernel(bool *label, vertex_data_type *vertexValue_in, vertex_data_type *vertexValue_out,
                                            const count_type vertexNum, const countl_type *csr_offset, const vertex_id_type *csr_dest)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);

    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertexNum)
    {
        if (vertexNum > chunkIdx)
            chunk_size = vertexNum - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        if (label[i])
        {
            const uint64_t start = csr_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            const uint64_t end = csr_offset[i + 1];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
                if (j >= start)
                    atomicAdd(&vertexValue_out[csr_dest[j]], vertexValue_in[i]);

            label[i] = false;
        }
    }
}

//> 1. [Coalesce]: coalesce_chunk_model
template <typename GraphDevice> void coalesce_chunk_model(GraphDevice &graphDevice, const count_type nBlock, const count_type nBlock_update)
{
    coalesce_chunk_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_device.in(),
                                                       graphDevice.vertexValue_device.out(), graphDevice.vertexNum, graphDevice.csr_offset_device,
                                                       graphDevice.csr_dest_device);

    update_kernel<<<nBlock_update, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_device.in(),
                                                graphDevice.vertexValue_device.out(), graphDevice.vertexNum, graphDevice.csr_offset_device,
                                                graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 1. [Coalesce]: coalesce_chunk_model_kernel
__global__ void coalesce_chunk_measure_model_kernel(vertex_data_type *vertexValue_in, vertex_data_type *vertexValue_out, const count_type vertexNum,
                                                    const countl_type *csr_offset, const vertex_id_type *csr_dest)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);

    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertexNum)
    {
        if (vertexNum > chunkIdx)
            chunk_size = vertexNum - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        const uint64_t start = csr_offset[i];
        const uint64_t shift_start = start & MEM_ALIGN_CPJ;
        const uint64_t end = csr_offset[i + 1];

        for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
            if (j >= start)
                atomicAdd(&vertexValue_out[csr_dest[j]], vertexValue_in[i]);
    }
}

//> 1. [Coalesce]: coalesce_chunk_measure_model
template <typename GraphDevice> void coalesce_chunk_measure_model(GraphDevice &graphDevice, const count_type nBlock, count_type vertexNumProcess)
{
    coalesce_chunk_measure_model_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.vertexValue_in_device, graphDevice.vertexValue_out_device,
                                                               vertexNumProcess, graphDevice.copResult.csrResult_device.csr_offset,
                                                               graphDevice.copResult.csrResult_device.csr_dest);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//> 1. [Coalesce]: coalesce_chunk_coProcess_model_kernel
__global__ void coalesce_chunk_coProcess_model_kernel(size_t *label, vertex_data_type *vertexValue_in, vertex_data_type *vertexValue_out,
                                                      const count_type vertexNum, const countl_type *csr_offset, const vertex_id_type *csr_dest)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64_t warpIdx = threadId >> WARP_SHIFT;
    const uint64_t laneIdx = threadId & ((1 << WARP_SHIFT) - 1);

    const uint64_t chunkIdx = warpIdx * CHUNK_SIZE_EMOGI;
    uint64_t chunk_size = CHUNK_SIZE_EMOGI;

    if ((chunkIdx + CHUNK_SIZE_EMOGI) > vertexNum)
    {
        if (vertexNum > chunkIdx)
            chunk_size = vertexNum - chunkIdx;
        else
            return;
    }

    for (uint32_t i = chunkIdx; i < chunk_size + chunkIdx; i++)
    {
        size_t arrpos = BIT_OFFSET(i);
        size_t bitpos = BIT_MOD(i);
        size_t one = (size_t(1) << size_t(bitpos));
        if (label[arrpos] & one)
        {
            const uint64_t start = csr_offset[i];
            const uint64_t shift_start = start & MEM_ALIGN_CPJ;
            const uint64_t end = csr_offset[i + 1];

            for (uint64_t j = shift_start + laneIdx; j < end; j += WARPSIZE)
                if (j >= start)
                    atomicAdd(&vertexValue_out[csr_dest[j]], vertexValue_in[i]);
        }
    }
}

//> 1. [Coalesce]: coalesce_chunk_model
template <typename GraphDevice> void coalesce_chunk_coProcess_model(GraphDevice &graphDevice, const count_type nBlock, count_type vertexNumProcess)
{
    coalesce_chunk_coProcess_model_kernel<<<nBlock, BLOCKSIZE>>>(
        graphDevice.label_cur_device, graphDevice.vertexValue_in_device, graphDevice.vertexValue_out_device, vertexNumProcess,
        graphDevice.copResult.csrResult_device.csr_offset, graphDevice.copResult.csrResult_device.csr_dest);

    // update_kernel<<<nBlock_update, BLOCKSIZE>>>(graphDevice.label_cur_device, graphDevice.vertexValue_device.in(),
    //                                             graphDevice.vertexValue_device.out(), graphDevice.vertexNum, graphDevice.csr_offset_device,
    //                                             graphDevice.common_device);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void update_coProcess_kernel(vertex_data_type *vertexValue_in, const count_type vertexNum)
{
    // const uint64_t tid = blockDim.x * BLOCK_SIZE * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    count_type threadId = threadIdx.x + blockIdx.x * blockDim.x;
    // if (threadId < vertexNum)
    //     printf("tid[%d] = %f\n", threadId, vertexValue_in[threadId]);
    if (threadId < vertexNum)
    {
        vertexValue_in[threadId] = 0.0f;
    }
}

template <typename GraphDevice> void update_coProcess_model(GraphDevice &graphDevice, const count_type nBlock)
{
    update_coProcess_kernel<<<nBlock, BLOCKSIZE>>>(graphDevice.vertexValue_in_device, graphDevice.vertexNum);

    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace PR_DEVICE_SPACE
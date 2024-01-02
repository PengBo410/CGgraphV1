#pragma once

#include "Basic/Console/console_V3.hpp"
#include "Basic/GPU/cuda_check.cuh"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Other/monitor.hpp"
#include "Basic/Type/data_type.hpp"
#include "Basic/basic_include.cuh"
#include "algorithm_device.hpp"
#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>

namespace Standalone {

class GraphDevice
{
  private:
    CSR_Result_type &csrResult;
    Algorithm_type algorithm;
    vertex_id_type root;
    std::string csrWeightFile = "";
    GPU_memory_type gpuMemory = GPU_memory_type::GPU_MEM;
    GPU_implement_type implement = GPU_implement_type::COALESCE_CHUNK;
    int useDeviceId = 0;

    Monitor *monitor = nullptr;
    static constexpr bool MONITOR = true;
    static constexpr bool DEBUG = true;
    uint64_t GPU_TOTAL_BYTE = 0;

    const int _threadNum = omp_get_max_threads();

  public:
    //^ CSR 事实证明这样真的能提高性能,而不是直接调用csrResult取数据
    count_type vertexNum;
    countl_type edgeNum;
    countl_type *csr_offset = nullptr;
    vertex_id_type *csr_dest = nullptr;
    edge_data_type *csr_weight = nullptr;
    degree_type *outDegree = nullptr;
    degree_type *inDegree = nullptr;
    countl_type *csr_offset_device = nullptr;
    vertex_id_type *csr_dest_device = nullptr;
    edge_data_type *csr_weight_device = nullptr;
    degree_type *outDegree_device = nullptr;
    degree_type *inDegree_device = nullptr;
    count_type *machine_offset = nullptr;

    //^ Active
    double worklistPercent = 0.1;
    vertex_id_type *worklist_in_device = nullptr;
    vertex_id_type *worklist_out_device = nullptr;
    countl_type worklistSize;
    DoubleBuffer_array<vertex_id_type> worklist_device;
    bool *label_cur_device = nullptr;
    bool *label_next_device = nullptr;
    DoubleBuffer_array<bool> label_device;

    //^ vertexValue
    vertex_data_type *vertexValue_in = nullptr;
    vertex_data_type *vertexValue_out = nullptr;
    vertex_data_type *vertexValue_result = nullptr;
    vertex_data_type *vertexValue_in_device = nullptr;
    vertex_data_type *vertexValue_out_device = nullptr;
    vertex_data_type *vertexValue_result_device = nullptr;
    DoubleBuffer_array<vertex_data_type> vertexValue_device;

    //^ common
    count_type common_size = 5;
    countl_type *common;
    countl_type *common_device;

  public:
    GraphDevice() = delete;
    GraphDevice(CSR_Result_type &csrResult_, GPU_memory_type gpuMemory_ = GPU_memory_type::GPU_MEM,
                GPU_implement_type implement_ = GPU_implement_type::COALESCE_CHUNK, int useDeviceId_ = 0)
        : csrResult(csrResult_), vertexNum(0), edgeNum(0), useDeviceId(useDeviceId_)
    {
        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("GraphDevice(), used max physical memory: %.2lf (GB)",
                         static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }

        CUDA_CHECK(cudaSetDevice(useDeviceId));

        vertexNum = csrResult_.vertexNum;
        edgeNum = csrResult_.edgeNum;
        csr_offset = csrResult_.csr_offset;
        csr_dest = csrResult_.csr_dest;
        csr_weight = csrResult_.csr_weight;
        outDegree = csrResult_.outDegree;
        inDegree = csrResult_.inDegree;

        root = std::numeric_limits<vertex_id_type>::max();
        gpuMemory = gpuMemory_;
        implement = implement_;

        worklistSize = worklistPercent * edgeNum;
        Msg_info("worklistSize = %zu, used %.2lf (GB) GPU Memory", SCU64(worklistSize), SCD(BYTES_TO_GB(worklistSize * sizeof(vertex_id_type) * 2)));

        //^ Malloc CPU/GPU Memory
        mallocMemory();

        //^ copy to GPU
        copyToGPU();
    }
    void setMonitor(Monitor *monitor_) { monitor = monitor_; }
    void setCsrWeightFile(std::string csrWeightFile_) { csrWeightFile = csrWeightFile_; }

    //> Graph Processing
  public:
    double graphProcess(Algorithm_type algorithm_, vertex_id_type root_)
    {
        algorithm = algorithm_;
        root = root_;
        bool isMaxIteCount = false;

        initMemoryForAlg();

        size_t freeMemory = 0;
        size_t totalMemory = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));
        Msg_info("The GPU[%d] total memory = %.2lf (GB), used memory = %.2lf (GB), free memory = %.2lf (GB)", useDeviceId,
                 BYTES_TO_GB(SCD(totalMemory)), BYTES_TO_GB(SCD(totalMemory - freeMemory)), BYTES_TO_GB(SCD(freeMemory)));

        int ite = 0;
        double processTime = 0.0;
        count_type nBlock = 0;
        count_type nBlock_update = 0;
        int level = 0;

        timer iteTime;
        timer singTime;
        do
        {
            ite++;
            singTime.start();

            switch (algorithm)
            {
            case Algorithm_type::BFS:
                switch (implement)
                {
                case GPU_implement_type::CAT:
                    common[1] = common[0];
                    common[0] = 0;
                    CUDA_CHECK(H2D(common_device, common, 2));
                    nBlock = (common[1] + BLOCKSIZE - 1) / BLOCKSIZE;
                    BFS_DEVICE_SPACE::CTA_imp::worklist_model(*this, nBlock);
                    worklist_device.swap();
                    break;
                case GPU_implement_type::COALESCE_CHUNK:
                    common[0] = 0;
                    CUDA_CHECK(H2D(common_device, common, 1));
                    nBlock = ((vertexNum * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                    BFS_DEVICE_SPACE::coalesce_chunk_model(*this, nBlock, level);
                    level++;
                    break;
                case GPU_implement_type::MAX_VALUE:
                    assert_msg(false, "You Appoint Error GPU_implement_type");
                    break;
                }
                break;
            case Algorithm_type::SSSP:
                switch (implement)
                {
                case GPU_implement_type::CAT:
                    common[1] = common[0];
                    common[0] = 0;
                    CUDA_CHECK(H2D(common_device, common, 2));
                    nBlock = (common[1] + BLOCKSIZE - 1) / BLOCKSIZE;
                    SSSP_DEVICE_SPACE::CTA_imp::worklist_model(*this, nBlock);
                    worklist_device.swap();
                    break;
                case GPU_implement_type::COALESCE_CHUNK:
                    common[0] = 0;
                    CUDA_CHECK(H2D(common_device, common, 1));
                    nBlock = ((vertexNum * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                    nBlock_update = ((vertexNum + BLOCKSIZE - 1) / BLOCKSIZE);
                    SSSP_DEVICE_SPACE::coalesce_chunk_model(*this, nBlock, nBlock_update);
                    break;
                case GPU_implement_type::MAX_VALUE:
                    assert_msg(false, "You Appoint Error GPU_implement_type");
                    break;
                }
                break;
            case Algorithm_type::WCC:
                switch (implement)
                {
                case GPU_implement_type::CAT:
                    common[1] = common[0];
                    common[0] = 0;
                    CUDA_CHECK(H2D(common_device, common, 2));
                    nBlock = (common[1] + BLOCKSIZE - 1) / BLOCKSIZE;
                    WCC_DEVICE_SPACE::CTA_imp::worklist_model(*this, nBlock);
                    worklist_device.swap();
                    break;
                case GPU_implement_type::COALESCE_CHUNK:
                    CUDA_CHECK(cudaMemset(label_device.out(), 0x00, vertexNum * sizeof(bool)));
                    common[0] = 0;
                    CUDA_CHECK(H2D(common_device, common, 1));
                    nBlock = ((vertexNum * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                    nBlock_update = ((vertexNum + BLOCKSIZE - 1) / BLOCKSIZE);
                    WCC_DEVICE_SPACE::coalesce_chunk_model(*this, nBlock);
                    label_device.swap();
                    break;
                case GPU_implement_type::MAX_VALUE:
                    assert_msg(false, "You Appoint Error GPU_implement_type");
                    break;
                }
                break;
            case Algorithm_type::PageRank:
                switch (implement)
                {
                case GPU_implement_type::CAT:
                    // 内部清零,无需再次归零
                    common[0] = 0;
                    CUDA_CHECK(H2D(common_device, common, 1));
                    nBlock = ((vertexNum * WARPSIZE + BLOCKSIZE - 1) / BLOCKSIZE);
                    nBlock_update = ((vertexNum + BLOCKSIZE - 1) / BLOCKSIZE);
                    PR_DEVICE_SPACE::coalesce_model(*this, nBlock, nBlock_update);
                    vertexValue_device.swap();
                    if (ite > root)
                        isMaxIteCount = true;
                    break;
                case GPU_implement_type::COALESCE_CHUNK:
                    // 内部清零,无需再次归零
                    common[0] = 0;
                    CUDA_CHECK(H2D(common_device, common, 1));
                    nBlock = ((vertexNum * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                    nBlock_update = ((vertexNum + BLOCKSIZE - 1) / BLOCKSIZE);
                    PR_DEVICE_SPACE::coalesce_chunk_model(*this, nBlock, nBlock_update);
                    vertexValue_device.swap();
                    if (ite > root)
                        isMaxIteCount = true;
                    break;
                case GPU_implement_type::MAX_VALUE:
                    assert_msg(false, "You Appoint Error GPU_implement_type");
                    break;
                }
                break;
            case Algorithm_type::MAX_VALUE:
                assert_msg(false, "You Appoint Error Algorithm_type");
                break;
            }

            CUDA_CHECK(D2H(common, common_device, 1)); // D2H
            if ((!common[0]) || isMaxIteCount)
            {
                processTime = iteTime.get_time_ms();
                break;
            }
        }
        while (true);

        // 为了结果检查我们将结果回传回Host
        CUDA_CHECK(D2H(vertexValue_result, vertexValue_result_device, vertexNum));
        // for (vertex_id_type vertexId = 0; vertexId < 7; vertexId++)
        // {
        //     std::cout << vertexValue_result[vertexId] << std::endl;
        //     // Msg_major("vertexId[%u] = %u", vertexId, vertexValue_result[vertexId]);
        // }

        return processTime;

    clear:
        clearGraph();
        STOP;
    }

  private:
    /**********************************************************************************************************
     * @description: Malloc The VertexValue_in , common, CSR for device, worklist
     **********************************************************************************************************/
    void mallocMemory()
    {
        timer malloc_time;

        //^ VertexValue_in
        CUDA_CHECK(MALLOC_HOST(&vertexValue_in, vertexNum));
        CUDA_CHECK(MALLOC_DEVICE(&vertexValue_in_device, vertexNum));
        if constexpr (DEBUG)
        {
            GPU_TOTAL_BYTE += (vertexNum * sizeof(vertex_data_type));
        }

        //^ common
        CUDA_CHECK(MALLOC_HOST(&common, common_size));
        CUDA_CHECK(MALLOC_DEVICE(&common_device, common_size));
        if constexpr (DEBUG)
        {
            GPU_TOTAL_BYTE += (common_size * sizeof(countl_type));
        }

        //^ CSR
        CUDA_CHECK(MALLOC_DEVICE(&csr_offset_device, (vertexNum + 1)));
        switch (gpuMemory)
        {
        case GPU_memory_type::GPU_MEM:
            CUDA_CHECK(MALLOC_DEVICE(&csr_dest_device, (edgeNum)));
            if constexpr (DEBUG)
            {
                GPU_TOTAL_BYTE += (edgeNum * sizeof(vertex_id_type));
            }
            break;
        case GPU_memory_type::UVM:
            CUDA_CHECK(cudaMallocManaged((void **)&csr_dest_device, edgeNum * sizeof(vertex_id_type)));
            CUDA_CHECK(cudaMemAdvise(csr_dest_device, edgeNum * sizeof(vertex_id_type), cudaMemAdviseSetReadMostly, useDeviceId));
            break;
        case GPU_memory_type::MAX_VALUE:
            assert_msg(false, "You Appoint Error GPU_memory_type");
            break;
        };
        if constexpr (DEBUG)
        {
            GPU_TOTAL_BYTE += ((vertexNum + 1) * sizeof(countl_type));
        }

        //^ worklist or active
        if (implement != GPU_implement_type::COALESCE_CHUNK)
        {
            CUDA_CHECK(MALLOC_DEVICE(&worklist_in_device, (worklistSize)));
            CUDA_CHECK(MALLOC_DEVICE(&worklist_out_device, (worklistSize)));
            worklist_device.setDoubleBuffer(worklist_in_device, worklist_out_device);
            if constexpr (DEBUG)
            {
                GPU_TOTAL_BYTE += ((worklistSize) * sizeof(countl_type) * 2);
            }
        }

        //^ machine_offset
        machine_offset = new count_type[2]; // This is standalone
        machine_offset[0] = 0;
        machine_offset[1] = vertexNum;

        Msg_info("Malloc CPU/GPU Memory used time: %.2lf (ms)", malloc_time.get_time_ms());

        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("Graph GPU mallocMemory, used max physical memory: %.2lf (GB)",
                         static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }
    }

    /**********************************************************************************************************
     * @description: Copy CSR and common To GPU
     * @return [*]
     **********************************************************************************************************/
    void copyToGPU()
    {
        timer copyToGPU_time;

        //^ CSR
        CUDA_CHECK(H2D(csr_offset_device, csr_offset, (vertexNum + 1)));
        switch (gpuMemory)
        {
        case GPU_memory_type::GPU_MEM:
            CUDA_CHECK(H2D(csr_dest_device, csr_dest, (edgeNum)));
            break;
        case GPU_memory_type::UVM:
            memcpy(csr_dest_device, csr_dest, edgeNum * sizeof(vertex_id_type));
            if (csr_dest != nullptr)
            {
                delete[] csr_dest;
                csr_dest = nullptr;
            }
            break;
        case GPU_memory_type::MAX_VALUE:
            assert_msg(false, "You Appoint Error GPU_memory_type");
            break;
        };

        //^ common
        countl_type *tmp = new countl_type[2];
        tmp[0] = static_cast<countl_type>(vertexNum);
        tmp[1] = static_cast<countl_type>(edgeNum);
        CUDA_CHECK(H2D(common_device + 2, tmp, 2));

        Msg_info("CopyToGPU used time: %.2lf (ms)", copyToGPU_time.get_time_ms());
    }

    /**********************************************************************************************************
     * @description:  Check Memory For The Specified  Algorithm
     **********************************************************************************************************/
    void checkMemoryForAlg()
    {
        switch (algorithm)
        {
        case Algorithm_type::BFS:
            if (csr_weight != nullptr)
            {
                delete[] csr_weight;
                csr_weight = nullptr;
            }
            if (csr_weight_device != nullptr)
            {
                CUDA_CHECK(FREE_DEVICE(csr_weight_device));
                csr_weight_device = nullptr;
            }
            if (vertexValue_out != nullptr)
            {
                CUDA_CHECK(FREE_HOST(vertexValue_out));
                vertexValue_out = nullptr;
            }
            break;
        case Algorithm_type::SSSP:
            if (csr_weight == nullptr)
            {
                if (csrWeightFile != "")
                {
                    csr_weight = load_binFile<edge_data_type>(csrWeightFile, static_cast<uint64_t>(edgeNum));
                }
                else
                {
                    assert_msg_clear(false, "You Ready Run %s, but you free [csr_weight] but do not appoint csrWeightFile",
                                     Algorithm_type_name[SCI32(algorithm)]);
                }
            }

            if (csr_weight_device == nullptr)
            {
                switch (gpuMemory)
                {
                case GPU_memory_type::GPU_MEM:
                    CUDA_CHECK(MALLOC_DEVICE(&csr_weight_device, edgeNum));
                    CUDA_CHECK(H2D(csr_weight_device, csr_weight, edgeNum));
                    if constexpr (DEBUG)
                    {
                        GPU_TOTAL_BYTE += (edgeNum * sizeof(vertex_id_type));
                    }
                    break;
                case GPU_memory_type::UVM:
                    CUDA_CHECK(cudaMallocManaged((void **)&csr_weight_device, edgeNum * sizeof(vertex_id_type)));
                    CUDA_CHECK(cudaMemAdvise(csr_weight_device, edgeNum * sizeof(vertex_id_type), cudaMemAdviseSetReadMostly, useDeviceId));
                    memcpy(csr_weight_device, csr_weight, sizeof(edge_data_type) * edgeNum);
                    if (csr_weight != nullptr)
                    {
                        delete[] csr_dest;
                        csr_weight = nullptr;
                    }
                    break;
                case GPU_memory_type::MAX_VALUE:
                    assert_msg(false, "You Appoint Error GPU_memory_type");
                    break;
                };
            }

            if (implement == GPU_implement_type::COALESCE_CHUNK)
            {
                CUDA_CHECK(MALLOC_DEVICE(&label_cur_device, vertexNum));
                if constexpr (DEBUG)
                {
                    GPU_TOTAL_BYTE += ((vertexNum) * sizeof(bool));
                }
                if (vertexValue_out == nullptr)
                {
                    CUDA_CHECK(MALLOC_DEVICE(&vertexValue_out_device, vertexNum));
                    if constexpr (DEBUG)
                    {
                        GPU_TOTAL_BYTE += (vertexNum * sizeof(vertex_data_type));
                    }
                }
            }
            else
            {
                if (vertexValue_out != nullptr)
                {
                    CUDA_CHECK(FREE_HOST(vertexValue_out));
                    vertexValue_out = nullptr;
                }
            }

            break;
        case Algorithm_type::WCC:
            if (csr_weight != nullptr)
            {
                delete[] csr_weight;
                csr_weight = nullptr;
            }
            if (csr_weight_device != nullptr)
            {
                CUDA_CHECK(FREE_DEVICE(csr_weight_device));
                csr_weight_device = nullptr;
            }
            if (vertexValue_out != nullptr)
            {
                CUDA_CHECK(FREE_HOST(vertexValue_out));
                vertexValue_out = nullptr;
            }
            if (implement == GPU_implement_type::COALESCE_CHUNK)
            {
                CUDA_CHECK(MALLOC_DEVICE(&label_cur_device, vertexNum));
                CUDA_CHECK(MALLOC_DEVICE(&label_next_device, vertexNum));
                label_device.setDoubleBuffer(label_cur_device, label_next_device);
                if constexpr (DEBUG)
                {
                    GPU_TOTAL_BYTE += (vertexNum * sizeof(bool) * 2);
                }
            }
            break;
        case Algorithm_type::PageRank:
            if (csr_weight != nullptr)
            {
                delete[] csr_weight;
                csr_weight = nullptr;
            }
            if (csr_weight_device != nullptr)
            {
                CUDA_CHECK(FREE_DEVICE(csr_weight_device));
                csr_weight_device = nullptr;
            }
            if (vertexValue_out == nullptr)
            {
                CUDA_CHECK(MALLOC_HOST(&vertexValue_out, vertexNum));
                CUDA_CHECK(MALLOC_DEVICE(&vertexValue_out_device, vertexNum));
                if constexpr (DEBUG)
                {
                    GPU_TOTAL_BYTE += (vertexNum * sizeof(edge_data_type));
                }
            }
            if (label_cur_device == nullptr)
            {
                CUDA_CHECK(MALLOC_DEVICE(&label_cur_device, vertexNum));
                if constexpr (DEBUG)
                {
                    GPU_TOTAL_BYTE += ((vertexNum) * sizeof(bool));
                }
            }

            break;
        case Algorithm_type::MAX_VALUE:
            assert_msg(false, "You Appoint Error Algorithm_type");
            break;
        }

        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Log_info("Graph checkMemoryForAlg, used max physical memory: %.2lf (GB)",
                         static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }
        return;

    clear:
        clearGraph();
        STOP;
    }

    /**********************************************************************************************************
     * @description:  Init Memory For The Specified  Algorithm
     **********************************************************************************************************/
    void initMemoryForAlg()
    {
        checkMemoryForAlg();

        switch (algorithm)
        {
        case Algorithm_type::BFS:
            // root 检查
            if (root == std::numeric_limits<vertex_id_type>::max())
                assert_msg_clear(false, "You Ready Run %s, but you do not appoint root", Algorithm_type_name[SCI32(algorithm)]);
            if (root >= vertexNum)
                assert_msg_clear(false, "You Ready Run %s, but you appoint a invalid root, root = %zu", Algorithm_type_name[SCI32(algorithm)],
                                 static_cast<uint64_t>(root));
            // 初始话顶点的值
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                vertexValue_in[vertexId] = std::numeric_limits<vertex_data_type>::max();
            }
            vertexValue_in[root] = 0;
            CUDA_CHECK(H2D(vertexValue_in_device, vertexValue_in, vertexNum));

            // active
            if (implement != GPU_implement_type::COALESCE_CHUNK)
            {
                CUDA_CHECK(H2D(worklist_device.in(), &root, 1));
            }

            // vertexValue_result
            vertexValue_result = vertexValue_in;
            vertexValue_result_device = vertexValue_in_device;
            // common
            common[0] = 1;

            break;

        case Algorithm_type::SSSP:
            // root 检查
            if (root == std::numeric_limits<vertex_id_type>::max())
                assert_msg_clear(false, "You Ready Run %s, but you do not appoint root", Algorithm_type_name[SCI32(algorithm)]);
            if (root >= vertexNum)
                assert_msg_clear(false, "You Ready Run %s, but you appoint a invalid root, root = %zu", Algorithm_type_name[SCI32(algorithm)],
                                 static_cast<uint64_t>(root));
            // 初始话顶点的值
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                vertexValue_in[vertexId] = std::numeric_limits<vertex_data_type>::max();
            }
            vertexValue_in[root] = 0;
            CUDA_CHECK(H2D(vertexValue_in_device, vertexValue_in, vertexNum));

            if (implement == GPU_implement_type::COALESCE_CHUNK)
            {
                CUDA_CHECK(H2D(vertexValue_out_device, vertexValue_in, vertexNum));
                CUDA_CHECK(cudaMemset(label_cur_device, 0x0, vertexNum * sizeof(bool)));
                bool one = true;
                CUDA_CHECK(H2D(&label_cur_device[root], &one, 1));
                // vertexValue_result
                vertexValue_result = vertexValue_in;                // 不用vertexValue_out的原因是vertexValue_out为nullptr
                vertexValue_result_device = vertexValue_out_device; // vertexValue_out_device -> costList
                common[0] = 1;
            }
            else
            {
                // active
                CUDA_CHECK(H2D(worklist_device.in(), &root, 1));
                // vertexValue_result
                vertexValue_result = vertexValue_in;
                vertexValue_result_device = vertexValue_in_device;
                // common
                common[0] = 1;
            }

            break;

        case Algorithm_type::WCC:
            // 初始话顶点的值
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++) { vertexValue_in[vertexId] = vertexId; }
            CUDA_CHECK(H2D(vertexValue_in_device, vertexValue_in, vertexNum));

            if (implement == GPU_implement_type::COALESCE_CHUNK)
            {
                CUDA_CHECK(cudaMemset(label_device.in(), 0x01, vertexNum * sizeof(bool))); // All Active
                CUDA_CHECK(cudaMemset(label_device.out(), 0x0, vertexNum * sizeof(bool)));
                common[0] = 1;
            }
            else
            {
                // active
                vertex_id_type *tmp = new vertex_id_type[vertexNum];
                omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++) { tmp[vertexId] = vertexId; }
                CUDA_CHECK(H2D(worklist_device.in(), tmp, vertexNum));
                delete[] tmp;
            }

            // vertexValue_result
            vertexValue_result = vertexValue_in;
            vertexValue_result_device = vertexValue_in_device;

            // common
            common[0] = vertexNum;

            break;

        case Algorithm_type::PageRank:
            // 检查root
            Msg_warn("You Ready Max Run %s, run iteration : %zu", Algorithm_type_name[SCI32(algorithm)], static_cast<uint64_t>(root));
            // 初始话顶点的值
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                // vertexValue_in[vertexId] = 1.0 - Alpha;
                // if (outDegree[vertexId] != 0)
                //     vertexValue_in[vertexId] = (1.0 - Alpha) * Alpha / outDegree[vertexId];
                // vertexValue_out[vertexId] = 0.0;

                vertexValue_in[vertexId] = 1.0;
                if (outDegree[vertexId] != 0)
                    vertexValue_in[vertexId] = 1.0 / outDegree[vertexId];
                vertexValue_out[vertexId] = 0.0;

                // Msg_info(" vertexValue_in[%u] = %f", vertexId, vertexValue_in[vertexId]);
            }
            CUDA_CHECK(H2D(vertexValue_in_device, vertexValue_in, vertexNum));
            CUDA_CHECK(H2D(vertexValue_out_device, vertexValue_out, vertexNum));
            vertexValue_device.setDoubleBuffer(vertexValue_in_device, vertexValue_out_device);

            CUDA_CHECK(cudaMemset(label_cur_device, 0x01, vertexNum * sizeof(bool))); // All Active

            // vertexValue_result
            vertexValue_result = vertexValue_in; // 我们多调用一个swap
            vertexValue_result_device = vertexValue_in_device;

            // common
            common[0] = vertexNum;

            break;

        case Algorithm_type::MAX_VALUE:
            assert_msg_clear(false, "You Not Appoint Suitable Algorithm");
            break;
        }

        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Log_info("Graph initMemoryForAlg, used max physical memory: %.2lf (GB)",
                         static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }
        return;

    clear:
        clearGraph();
        STOP;
    }

  public:
    void clearGraph()
    {
        csrResult.clearCSR();
        if (csr_offset_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_offset_device));
            csr_offset_device = nullptr;
        }
        if (csr_dest_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_dest_device));
            csr_dest_device = nullptr;
        }
        if (csr_weight_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(csr_weight_device));
            csr_weight_device = nullptr;
        }
        if (outDegree_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(outDegree_device));
            outDegree_device = nullptr;
        }
        if (inDegree_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(inDegree_device));
            inDegree_device = nullptr;
        }
        if (common != nullptr)
        {
            CUDA_CHECK(FREE_HOST(common));
            common = nullptr;
        }
        if (common_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(common_device));
            common_device = nullptr;
        }

        if (worklist_in_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(worklist_in_device));
            worklist_in_device = nullptr;
        }

        if (worklist_out_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(worklist_out_device));
            worklist_out_device = nullptr;
        }

        if (vertexValue_in != nullptr)
        {
            CUDA_CHECK(FREE_HOST(vertexValue_in));
            vertexValue_in = nullptr;
        }

        if (vertexValue_in_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(vertexValue_in_device));
            vertexValue_in_device = nullptr;
        }

        if (vertexValue_out != nullptr)
        {
            CUDA_CHECK(FREE_HOST(vertexValue_out));
            vertexValue_out = nullptr;
        }

        if (vertexValue_out_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(vertexValue_out_device));
            vertexValue_out_device = nullptr;
        }

        if (machine_offset != nullptr)
        {
            delete[] machine_offset;
            machine_offset = nullptr;
        }

        // if constexpr (MONITOR)
        // {
        //     if (monitor != nullptr)
        //     {
        //         Msg_info("~Graph(), used max physical memory: %.2lf (GB)", static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 /
        //         1024);
        //     }
        // }
    }

}; // end of class [GraphDevice]

} // namespace Standalone
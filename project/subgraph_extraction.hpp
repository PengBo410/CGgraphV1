#pragma once

#include "Basic/Balance/taskSteal.hpp"
#include "Basic/Console/console_V3.hpp"
#include "Basic/GPU/cuda_check.cuh"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/Type/data_type.hpp"
#include "Basic/basic_include.cuh"

class SubgraphExtraction
{

  private:
    Monitor *monitor = nullptr;
    static constexpr bool MONITOR = true;
    static constexpr bool DEBUG = true;

  public:
    CSR_Result_type &csrResult;
    size_t GPUMemorySize;
    Algorithm_type algorithm = Algorithm_type::SSSP;
    int useDeviceId = 0;
    size_t curEdge = 0;
    GPU_memory_type GPUMemory;

    Coprocess_result_type copResult;

    SubgraphExtraction(CSR_Result_type &csrResult_, size_t GPUMemorySize_, Algorithm_type algorithm_, GPU_memory_type GPUMemory_,
                       int useDeviceId_ = 0)
        : csrResult(csrResult_), GPUMemorySize(GPUMemorySize_), algorithm(algorithm_), GPUMemory(GPUMemory_), useDeviceId(useDeviceId_)
    {

        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("Graph(), used max physical memory: %.2lf (GB)", SCD(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }
        GPUMemory = GPU_memory_type::GPU_MEM; // Current Only Support GPU_memory_type::GPU_MEM
        // Msg_major("SubgraphExtraction GPUMemory = %s", GPU_memory_type_name[SCI32(GPUMemory)]);

        changeToZeroCopy();

        subgraph_extraction_approximate();
        // subgraph_extraction_accurate();

        // if (!copResult.isDeviceWhole)
        // {
        //     buildOwnCPU_csrOffset();
        // }

        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("SubgraphExtraction(), used max physical memory: %.2lf (GB)", SCD(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }

        size_t freeMemory = 0;
        size_t totalMemory = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));
        Msg_info("The GPU[%d] total memory = %.2lf (GB), used memory = %.2lf (GB), free memory = %.2lf (GB)", useDeviceId,
                 BYTES_TO_GB(SCD(totalMemory)), BYTES_TO_GB(SCD(totalMemory - freeMemory)), BYTES_TO_GB(SCD(freeMemory)));
    }
    void setMonitor(Monitor *monitor_) { monitor = monitor_; }
    Coprocess_result_type getCoProcessResult() { return copResult; }

  private:
    void changeToZeroCopy()
    {
        timer t;

        copResult.csrResult_host.vertexNum = csrResult.vertexNum;
        copResult.csrResult_host.edgeNum = csrResult.edgeNum;
        copResult.csrResult_host.noZeroOutDegreeNum = csrResult.noZeroOutDegreeNum;

        countl_type *csr_offset_tmp;
        CUDA_CHECK(cudaMallocManaged((void **)&csr_offset_tmp, (csrResult.vertexNum + 1) * sizeof(countl_type)));
        CUDA_CHECK(cudaMemAdvise(csr_offset_tmp, (csrResult.vertexNum + 1) * sizeof(countl_type), cudaMemAdviseSetAccessedBy, useDeviceId));
        memcpy(csr_offset_tmp, csrResult.csr_offset, (csrResult.vertexNum + 1) * sizeof(countl_type));
        delete[] csrResult.csr_offset;
        csrResult.csr_offset = nullptr;
        copResult.csrResult_host.csr_offset = csr_offset_tmp;

        vertex_id_type *csr_dest_tmp;
        CUDA_CHECK(cudaMallocManaged((void **)&csr_dest_tmp, (csrResult.edgeNum) * sizeof(vertex_id_type)));
        CUDA_CHECK(cudaMemAdvise(csr_dest_tmp, (csrResult.edgeNum) * sizeof(vertex_id_type), cudaMemAdviseSetAccessedBy, useDeviceId));
        memcpy(csr_dest_tmp, csrResult.csr_dest, (csrResult.edgeNum) * sizeof(vertex_id_type));
        delete[] csrResult.csr_dest;
        csrResult.csr_dest = nullptr;
        copResult.csrResult_host.csr_dest = csr_dest_tmp;

        degree_type *outDeegree_tmp;
        CUDA_CHECK(cudaMallocManaged((void **)&outDeegree_tmp, (csrResult.vertexNum) * sizeof(degree_type)));
        CUDA_CHECK(cudaMemAdvise(outDeegree_tmp, (csrResult.vertexNum) * sizeof(degree_type), cudaMemAdviseSetAccessedBy, useDeviceId));
        memcpy(outDeegree_tmp, csrResult.outDegree, (csrResult.vertexNum) * sizeof(degree_type));
        delete[] csrResult.outDegree;
        csrResult.outDegree = nullptr;
        copResult.csrResult_host.outDegree = outDeegree_tmp;

        if (algorithm == Algorithm_type::SSSP)
        {
            edge_data_type *csr_weight_tmp;
            CUDA_CHECK(cudaMallocManaged((void **)&csr_weight_tmp, (csrResult.edgeNum) * sizeof(edge_data_type)));
            CUDA_CHECK(cudaMemAdvise(csr_weight_tmp, (csrResult.edgeNum) * sizeof(edge_data_type), cudaMemAdviseSetAccessedBy, useDeviceId));
            memcpy(csr_weight_tmp, csrResult.csr_weight, (csrResult.edgeNum) * sizeof(edge_data_type));
            delete[] csrResult.csr_weight;
            csrResult.csr_weight = nullptr;
            copResult.csrResult_host.csr_weight = csr_weight_tmp;
        }

        Msg_info("SubgraphExtraction - changeToZeroCopy finished, Used time: %.2lf (ms)", t.get_time_ms());
    }

    /****************************************************************************************************************************************
     * 返回提取的subgraph的CSR
     ****************************************************************************************************************************************/
    void build_subgraph(degree_type *outDegree, vertex_id_type cutVertexId, countl_type cutEdge)
    {
        CSR_Result_type csrResult_GPU; // 存储GPU端的子图
        csrResult_GPU.vertexNum = cutVertexId + 1;
        csrResult_GPU.edgeNum = cutEdge;
        csrResult_GPU.csr_offset = new countl_type[csrResult_GPU.vertexNum + 1];
        csrResult_GPU.csr_dest = new vertex_id_type[csrResult_GPU.edgeNum];
        if (algorithm == Algorithm_type::SSSP)
        {
            csrResult_GPU.csr_weight = new edge_data_type[csrResult_GPU.edgeNum];
        }

        degree_type *outDegree_GPU = new degree_type[csrResult_GPU.vertexNum];
        memset(outDegree_GPU, 0, sizeof(degree_type) * (csrResult_GPU.vertexNum));

        Balance::TaskSteal *taskSteal = new Balance::TaskSteal();

        timer build_time;
        timer build_time_tmp;

        build_time_tmp.start();
        taskSteal->taskSteal_twoStage<size_t, size_t>(
            static_cast<size_t>(csrResult_GPU.vertexNum),
            [&](size_t &current, size_t &local_workloads)
            {
            size_t end = current + VERTEXWORK_CHUNK;
            size_t length = VERTEXWORK_CHUNK;
            if (end >= csrResult_GPU.vertexNum)
                length = csrResult_GPU.vertexNum - current;

            for (size_t in = 0; in < length; in++)
            {
                vertex_id_type vertexId = current + in;

                // 二分查找填充outDegree_GPU
                countl_type offset = copResult.csrResult_host.csr_offset[vertexId];
                std::pair<bool, int64_t> index = searchEleNum(&copResult.csrResult_host.csr_dest[offset], outDegree[vertexId], cutVertexId);
                if (index.first)
                    outDegree_GPU[vertexId] = index.second + 1;
                else
                    outDegree_GPU[vertexId] = index.second;
            }
            },
            VERTEXWORK_CHUNK);
        csrResult_GPU.outDegree = outDegree_GPU;
        Msg_info("Step1: build outDegree_GPU complete, time = %.2f (ms)", build_time_tmp.get_time_ms());

        build_time_tmp.start();
        csrResult_GPU.csr_offset[0] = 0;
        for (vertex_id_type vertexId = 1; vertexId <= csrResult_GPU.vertexNum; vertexId++)
        {
            csrResult_GPU.csr_offset[vertexId] = csrResult_GPU.csr_offset[vertexId - 1] + static_cast<countl_type>(outDegree_GPU[vertexId - 1]);
        }
        Msg_info("Step2: build csr_offset_GPU complete, time = %.2f (ms)", build_time_tmp.get_time_ms());

        build_time_tmp.start();
        taskSteal->taskSteal_twoStage<size_t, size_t>(
            static_cast<size_t>(csrResult_GPU.vertexNum),
            [&](size_t &current, size_t &local_workloads)
            {
            size_t end = current + VERTEXWORK_CHUNK;
            size_t length = VERTEXWORK_CHUNK;
            if (end >= csrResult_GPU.vertexNum)
                length = csrResult_GPU.vertexNum - current;

            for (size_t in = 0; in < length; in++)
            {
                vertex_id_type vertexId = current + in;
                memcpy(csrResult_GPU.csr_dest + csrResult_GPU.csr_offset[vertexId],
                       copResult.csrResult_host.csr_dest + copResult.csrResult_host.csr_offset[vertexId],
                       outDegree_GPU[vertexId] * sizeof(vertex_id_type));

                if (algorithm == Algorithm_type::SSSP)
                {
                    memcpy(csrResult_GPU.csr_weight + csrResult_GPU.csr_offset[vertexId],
                           copResult.csrResult_host.csr_weight + copResult.csrResult_host.csr_offset[vertexId],
                           outDegree_GPU[vertexId] * sizeof(vertex_data_type));
                }
            }
            },
            VERTEXWORK_CHUNK);

        degree_type *outDegree_tmp = new degree_type[csrResult_GPU.vertexNum];
        memset(outDegree_tmp, 0, csrResult_GPU.vertexNum * sizeof(degree_type));
        omp_parallel_for(vertex_id_type vertexId = 0; vertexId < csrResult_GPU.vertexNum; vertexId++)
        {
            outDegree_tmp[vertexId] = csrResult_GPU.csr_offset[vertexId + 1] - csrResult_GPU.csr_offset[vertexId];
        }
        Msg_info("Step3: build csr_GPU complete, time = %.2f (ms)", build_time_tmp.get_time_ms());

        copResult.isDeviceWhole = false;
        copResult.csrResult_device.vertexNum = csrResult_GPU.vertexNum;
        copResult.csrResult_device.edgeNum = csrResult_GPU.edgeNum;
        copResult.csrResult_device.noZeroOutDegreeNum = csrResult_GPU.noZeroOutDegreeNum;
        copResult.csrResult_device.outDegree = outDegree_tmp; // which is a host ptr
        copResult.cutVertex = cutVertexId;
        CUDA_CHECK(MALLOC_DEVICE(&copResult.csrResult_device.csr_offset, csrResult_GPU.vertexNum + 1));
        CUDA_CHECK(H2D(copResult.csrResult_device.csr_offset, csrResult_GPU.csr_offset, csrResult_GPU.vertexNum + 1));

        switch (GPUMemory)
        {
        case GPU_memory_type::GPU_MEM:
            CUDA_CHECK(MALLOC_DEVICE(&copResult.csrResult_device.csr_dest, csrResult_GPU.edgeNum));
            break;
        case GPU_memory_type::UVM:
            CUDA_CHECK(cudaMallocManaged((void **)&copResult.csrResult_device.csr_dest, (csrResult_GPU.edgeNum) * sizeof(degree_type)));
            CUDA_CHECK(cudaMemAdvise(copResult.csrResult_device.csr_dest, (csrResult_GPU.edgeNum) * sizeof(vertex_id_type),
                                     cudaMemAdviseSetReadMostly, useDeviceId));
            break;
        case GPU_memory_type::MAX_VALUE:
            break;
        }
        CUDA_CHECK(H2D(copResult.csrResult_device.csr_dest, csrResult_GPU.csr_dest, csrResult_GPU.edgeNum));
        if (algorithm == Algorithm_type::SSSP)
        {
            switch (GPUMemory)
            {
            case GPU_memory_type::GPU_MEM:
                CUDA_CHECK(MALLOC_DEVICE(&copResult.csrResult_device.csr_weight, csrResult_GPU.edgeNum));
                break;
            case GPU_memory_type::UVM:
                CUDA_CHECK(cudaMallocManaged((void **)&copResult.csrResult_device.csr_weight, (csrResult_GPU.edgeNum) * sizeof(degree_type)));
                CUDA_CHECK(cudaMemAdvise(copResult.csrResult_device.csr_weight, (csrResult_GPU.edgeNum) * sizeof(vertex_id_type),
                                         cudaMemAdviseSetReadMostly, useDeviceId));
                break;
            case GPU_memory_type::MAX_VALUE:
                break;
            }

            CUDA_CHECK(H2D(copResult.csrResult_device.csr_weight, csrResult_GPU.csr_weight, csrResult_GPU.edgeNum));
        }

        csrResult_GPU.clearCSR();
        Msg_info("Build GPU CSR complete, time = %.2f (ms)", build_time.get_time_ms());
    }

    /****************************************************************************************************************************************
     * 返回提取的subgraph的CSR
     * 注意：这一版本中我们并不会占用100%的GPU内存，但性能好。换言之，我们只抽取了左上角的第一块正方形
     *
     * @param [csrResult]        整图的CSR
     * @param [GPUMemorySize]    GPU可用的memory大小
     * @param [algorithm]        algorithm为SSSP时，边的大小需要额外存储edge data
     *
     * @return 提取出的满足GPU内存大小的子图的CSR
     ****************************************************************************************************************************************/
    void subgraph_extraction_approximate()
    {
        degree_type *outDegree = copResult.csrResult_host.outDegree;

        size_t edgeSize = 0;
        if (algorithm == Algorithm_type::SSSP)
            edgeSize = sizeof(edge_data_type) + sizeof(vertex_id_type);
        else
            edgeSize = sizeof(vertex_id_type);

        assert_msg(GPUMemorySize > 2 * edgeSize, "You GPU Memory Size Input Error");
        Msg_info("csrResult.noZeroOutDegreeNum = %u", csrResult.noZeroOutDegreeNum);

        size_t totalEdgeSize = csrResult.edgeNum * edgeSize;
        if (totalEdgeSize <= GPUMemorySize)
        {
            Msg_info("All vertices will be extracted to GPU");
            copResult.csrResult_device.vertexNum = csrResult.vertexNum;
            copResult.csrResult_device.edgeNum = csrResult.edgeNum;
            copResult.csrResult_device.noZeroOutDegreeNum = csrResult.noZeroOutDegreeNum;
            copResult.csrResult_device.outDegree = copResult.csrResult_host.outDegree;
            copResult.cutVertex = csrResult.noZeroOutDegreeNum - 1; //!
            copResult.isDeviceWhole = true;

            countl_type *csr_offset_tmp;
            CUDA_CHECK(MALLOC_DEVICE(&csr_offset_tmp, (csrResult.vertexNum + 1)));
            CUDA_CHECK(H2D(csr_offset_tmp, copResult.csrResult_host.csr_offset, (csrResult.vertexNum + 1)));
            copResult.csrResult_device.csr_offset = csr_offset_tmp;

            vertex_id_type *csr_dest_tmp;
            CUDA_CHECK(MALLOC_DEVICE(&csr_dest_tmp, (csrResult.edgeNum)));
            CUDA_CHECK(H2D(csr_dest_tmp, copResult.csrResult_host.csr_dest, (csrResult.edgeNum)));
            copResult.csrResult_device.csr_dest = csr_dest_tmp;

            if (algorithm == Algorithm_type::SSSP)
            {
                edge_data_type *csr_weight_tmp;
                CUDA_CHECK(MALLOC_DEVICE(&csr_weight_tmp, (csrResult.edgeNum)));
                CUDA_CHECK(H2D(csr_weight_tmp, copResult.csrResult_host.csr_weight, (csrResult.edgeNum)));
                copResult.csrResult_device.csr_weight = csr_weight_tmp;
            }

            return;
        }
        else
        {
            Msg_info("Graph Size > GPU Memory Size, biginning extracting the subgraph");
            timer extra_time;
            timer tmp_time;

            size_t edgeNum_GPU = GPUMemorySize / edgeSize;
            Msg_info("Current GPU can store (%zu) edges, GPUMemorySize = %zu, edgeSize = %zu", edgeNum_GPU, GPUMemorySize, edgeSize);

            // Step1: 我们首先寻找下界
            tmp_time.start();
            vertex_id_type lower = 0;
            curEdge = 0;
            for (vertex_id_type vertexId = 0; vertexId < csrResult.noZeroOutDegreeNum; vertexId++)
            {
                curEdge += outDegree[vertexId];
                if (curEdge > edgeNum_GPU)
                {
                    lower = vertexId - 1;
                    break;
                }
            }
            assert_msg(lower < csrResult.noZeroOutDegreeNum, "lower computing error, lower = %zu", SCU64(lower));
            Msg_info("Step1: vertexId-lower = %zu, time = %.2f (ms)", SCU64(lower), tmp_time.get_time_ms());

            // Step2: 我们计算 [lower - lower]的边的个数
            tmp_time.start();
            curEdge = 0;
#pragma omp parallel for reduction(+ : curEdge)
            for (vertex_id_type vertexId = 0; vertexId <= lower; vertexId++)
            {
                countl_type offset = copResult.csrResult_host.csr_offset[vertexId];
                assert_msg((copResult.csrResult_host.csr_offset[vertexId + 1] - offset) == outDegree[vertexId],
                           "(copResult.csrResult_host.csr_offset[vertexId + 1] - offset) != outDegree[vertexId]");
                // 二分查找
                std::pair<bool, int64_t> index = searchEleNum(&copResult.csrResult_host.csr_dest[offset], outDegree[vertexId], lower);
                assert_msg(((index.second >= 0) && (index.second <= outDegree[vertexId])), "Bin search error, index = %zu",
                           static_cast<size_t>(index.second));
                if (index.first)
                {
                    curEdge += (index.second + 1);
                }
                else
                {
                    curEdge += index.second;
                }
            }
            assert_msg((curEdge) <= edgeNum_GPU, "curEdge > edgeNum_GPU");
            Msg_info("Step2: [%zu * %zu]共包含(%zu)条边, 占用(%.2f)%%的GPU内存, time = %.2f (ms)", static_cast<size_t>(lower),
                     static_cast<size_t>(lower), curEdge, ((double)curEdge * edgeSize) / GPUMemorySize * 100, tmp_time.get_time_ms());

            // Step3: Build GPU CSR
            tmp_time.start();
            build_subgraph(outDegree, lower, curEdge);
            Msg_info("Step3: Build GPU CSR complete, time = %.2f (ms)", tmp_time.get_time_ms());

            Msg_info("Total time = %.2f (ms)", extra_time.get_time_ms());

        } // end of if [totalEdgeSize > GPUMemorySize]
    }

    void subgraph_extraction_accurate()
    {
        degree_type *outDegree = copResult.csrResult_host.outDegree;

        size_t edgeSize = 0;
        if (algorithm == Algorithm_type::SSSP)
            edgeSize = sizeof(edge_data_type) + sizeof(vertex_id_type);
        else
            edgeSize = sizeof(vertex_id_type);

        assert_msg(GPUMemorySize > 2 * edgeSize, "You GPU Memory Size Input Error");
        Msg_info("csrResult.noZeroOutDegreeNum = %u", csrResult.noZeroOutDegreeNum);

        size_t totalEdgeSize = csrResult.edgeNum * edgeSize;
        if (totalEdgeSize <= GPUMemorySize)
        {
            Msg_info("All vertices will be extracted to GPU");
            copResult.csrResult_device.vertexNum = csrResult.vertexNum;
            copResult.csrResult_device.edgeNum = csrResult.edgeNum;
            copResult.csrResult_device.noZeroOutDegreeNum = csrResult.noZeroOutDegreeNum;
            copResult.csrResult_device.outDegree = copResult.csrResult_host.outDegree;
            copResult.cutVertex = csrResult.noZeroOutDegreeNum - 1; //!
            copResult.isDeviceWhole = true;

            countl_type *csr_offset_tmp;
            CUDA_CHECK(MALLOC_DEVICE(&csr_offset_tmp, (csrResult.vertexNum + 1)));
            CUDA_CHECK(H2D(csr_offset_tmp, copResult.csrResult_host.csr_offset, (csrResult.vertexNum + 1)));
            copResult.csrResult_device.csr_offset = csr_offset_tmp;

            vertex_id_type *csr_dest_tmp;
            CUDA_CHECK(MALLOC_DEVICE(&csr_dest_tmp, (csrResult.edgeNum)));
            CUDA_CHECK(H2D(csr_dest_tmp, copResult.csrResult_host.csr_dest, (csrResult.edgeNum)));
            copResult.csrResult_device.csr_dest = csr_dest_tmp;

            if (algorithm == Algorithm_type::SSSP)
            {
                edge_data_type *csr_weight_tmp;
                CUDA_CHECK(MALLOC_DEVICE(&csr_weight_tmp, (csrResult.edgeNum)));
                CUDA_CHECK(H2D(csr_weight_tmp, copResult.csrResult_host.csr_weight, (csrResult.edgeNum)));
                copResult.csrResult_device.csr_weight = csr_weight_tmp;
            }

            return;
        }
        else
        {
            Msg_info("Graph Size > GPU Memory Size, biginning extracting the subgraph");
            timer extra_time;
            timer tmp_time;

            size_t edgeNum_GPU = GPUMemorySize / edgeSize;
            Msg_info("Current GPU can store (%zu) edges, GPUMemorySize = %zu, edgeSize = %zu", edgeNum_GPU, GPUMemorySize, edgeSize);

            // Step1: 我们首先寻找下界
            tmp_time.start();
            vertex_id_type lower = 0;
            curEdge = 0;
            for (vertex_id_type vertexId = 0; vertexId < csrResult.noZeroOutDegreeNum; vertexId++)
            {
                curEdge += outDegree[vertexId];
                if (curEdge > edgeNum_GPU)
                {
                    lower = vertexId - 1;
                    break;
                }
            }
            assert_msg(lower < csrResult.noZeroOutDegreeNum, "lower computing error, lower = %zu", SCU64(lower));
            Msg_info("Step1: vertexId-lower = %zu, time = %.2f (ms)", SCU64(lower), tmp_time.get_time_ms());

            // Step2: 我们计算 [lower - lower]的边的个数
            tmp_time.start();
            curEdge = 0;
#pragma omp parallel for reduction(+ : curEdge)
            for (vertex_id_type vertexId = 0; vertexId <= lower; vertexId++)
            {
                countl_type offset = copResult.csrResult_host.csr_offset[vertexId];
                assert_msg((copResult.csrResult_host.csr_offset[vertexId + 1] - offset) == outDegree[vertexId],
                           "(copResult.csrResult_host.csr_offset[vertexId + 1] - offset) != outDegree[vertexId]");
                // 二分查找
                std::pair<bool, int64_t> index = searchEleNum(&copResult.csrResult_host.csr_dest[offset], outDegree[vertexId], lower);
                assert_msg(((index.second >= 0) && (index.second <= outDegree[vertexId])), "Bin search error, index = %zu",
                           static_cast<size_t>(index.second));
                if (index.first)
                {
                    curEdge += (index.second + 1);
                }
                else
                {
                    curEdge += index.second;
                }
            }
            assert_msg((curEdge) <= edgeNum_GPU, "curEdge > edgeNum_GPU");
            Msg_info("Step2: [%zu * %zu]共包含(%zu)条边, 占用(%.2f)%%的GPU内存, time = %.2f (ms)", static_cast<size_t>(lower),
                     static_cast<size_t>(lower), curEdge, ((double)curEdge * edgeSize) / GPUMemorySize * 100, tmp_time.get_time_ms());

            //> 添加
            bool estimate_upper = false; // 可以提高性能
            vertex_id_type upper = 0;
            if (estimate_upper)
            {
                // Step3: 我们计算 upper
                tmp_time.start();
                countl_type resultEdgeNum = edgeNum_GPU - curEdge;
                Msg_info("Step3: resultEdgeNum = %u", resultEdgeNum);
                countl_type resultEdgeNum_tmp = 0;
                upper = 0;
                for (vertex_id_type vertexId = (lower + 1); vertexId < csrResult.noZeroOutDegreeNum; vertexId++)
                {
                    resultEdgeNum_tmp += outDegree[vertexId];
                    if (resultEdgeNum_tmp > resultEdgeNum)
                    {
                        upper = vertexId - 1;
                        break;
                    }
                    if (vertexId == (csrResult.noZeroOutDegreeNum - 1))
                    {
                        upper = vertexId;
                        Msg_info("upper to veretxId, upper = %u", upper);
                    }
                }
                Msg_info("Vertex: (lower, upper] = (%zu, %zu], time = %.2f (ms)", static_cast<size_t>(lower), static_cast<size_t>(upper),
                         tmp_time.get_time_ms());

                // Step4: 此处我们应该验证下upper是否为真正的upper
                tmp_time.start();
                if (upper != (csrResult.noZeroOutDegreeNum - 1))
                {

                    while (true)
                    {
                        curEdge = 0;
#pragma omp parallel for reduction(+ : curEdge)
                        for (vertex_id_type vertexId = 0; vertexId <= upper; vertexId++)
                        {
                            countl_type offset = copResult.csrResult_host.csr_offset[vertexId];
                            assert_msg(
                                (copResult.csrResult_host.csr_offset[vertexId + 1] - offset) == outDegree[vertexId],
                                "(csrResult.csr_offset[vertexId + 1] - offset) != outDegree[vertexId], vertexId = %u, offset = %u, degree = %u",
                                vertexId, (copResult.csrResult_host.csr_offset[vertexId + 1] - offset), outDegree[vertexId]);
                            // 二分查找
                            std::pair<bool, int64_t> index = searchEleNum(&copResult.csrResult_host.csr_dest[offset], outDegree[vertexId], upper);
                            assert_msg(((index.second >= 0) && (index.second <= outDegree[vertexId])), "Bin search error, index = %zu",
                                       static_cast<size_t>(index.second));
                            if (index.first)
                                curEdge += (index.second + 1);
                            else
                                curEdge += index.second;
                        }

                        size_t inc = 6400;
                        if ((curEdge) >= edgeNum_GPU)
                            break;
                        else
                        {
                            upper = ((upper + inc) < csrResult.noZeroOutDegreeNum) ? (upper + inc) : (csrResult.noZeroOutDegreeNum - 1);
                        }
                    }

                    assert_msg((curEdge) >= edgeNum_GPU, "upper 的 curEdge < edgeNum_GPU");
                }
                Msg_info("Step4: time = %.2f (ms)", tmp_time.get_time_ms());
            }
            else
            {
                upper = csrResult.noZeroOutDegreeNum - 1;
            }

            // Step5: 在(lower, upper)间寻找满足lastVertexId
            tmp_time.start();
            uint64_t ite_binSearch = 0;
            vertex_id_type start = lower + 1;
            vertex_id_type end = upper;
            size_t curEdge_ = 0;
            vertex_id_type lastVertexId = 0;
            while (start <= end)
            {
                ite_binSearch++;
                vertex_id_type mid = (end - start) / 2 + start;
                curEdge_ = 0; // 每次迭代都归零
                              // Msg_info("mid == %u", mid);

                // 二分查找

#pragma omp parallel for reduction(+ : curEdge_)
                for (vertex_id_type vertexId = 0; vertexId <= mid; vertexId++)
                {
                    countl_type offset = copResult.csrResult_host.csr_offset[vertexId];
                    // 二分查找
                    std::pair<bool, int64_t> index = searchEleNum(&copResult.csrResult_host.csr_dest[offset], outDegree[vertexId], mid);
                    assert_msg(((index.second >= 0) && (index.second <= outDegree[vertexId])), "Bin search error, index = %zu",
                               static_cast<size_t>(index.second));
                    if (index.first)
                        curEdge_ += (index.second + 1);
                    else
                        curEdge_ += index.second;

                    // Msg_info("vertexId = %u, curEdge_ = %zu", vertexId, curEdge_);
                }
                // Msg_info("mid = %u, curEdge_ = %u", mid, curEdge_);

                if (edgeNum_GPU == curEdge_)
                {
                    lastVertexId = mid;
                    break;
                }
                else if (edgeNum_GPU > curEdge_)
                {
                    lastVertexId = mid;
                    start = mid + 1;
                }
                else if (edgeNum_GPU < curEdge_)
                {
                    end = mid - 1;
                    lastVertexId = end;
                }
            }
            Msg_info("Step5: time = %.2f (ms)", tmp_time.get_time_ms());

            Msg_info("End extracting the subgraph,  ite_binSearch = %lu, lastVertexId = %u, time = %.2f (ms)", ite_binSearch, lastVertexId,
                     extra_time.get_time_ms());
            Msg_info("Step5: [%zu * %zu]共包含(%zu)条边, 占用(%.2f)%%的GPU内存, time = %.2f (ms)", static_cast<size_t>(lastVertexId),
                     static_cast<size_t>(lastVertexId), curEdge_, ((double)curEdge_ * edgeSize) / GPUMemorySize * 100, tmp_time.get_time_ms());

            //> ----
            // Step3: Build GPU CSR
            tmp_time.start();
            build_subgraph(outDegree, lastVertexId, curEdge_);
            // build_subgraph(outDegree, lower, curEdge); //近似
            Msg_info("Step6: Build GPU CSR complete, time = %.2f (ms)", tmp_time.get_time_ms());

            Msg_info("Total time = %.2f (ms)", extra_time.get_time_ms());

        } // end of if [totalEdgeSize > GPUMemorySize]
    }

    void buildOwnCPU_csrOffset()
    {
        timer t;
        countl_type *csr_offset_tmp;
        CUDA_CHECK(cudaMallocManaged((void **)&csr_offset_tmp, (copResult.cutVertex + 1 + 1) * sizeof(countl_type)));
        CUDA_CHECK(cudaMemAdvise(csr_offset_tmp, (csrResult.vertexNum + 1) * sizeof(countl_type), cudaMemAdviseSetAccessedBy, useDeviceId));
        csr_offset_tmp[0] = 0;
        for (countl_type vertexId = 1; vertexId < (copResult.cutVertex + 1 + 1); vertexId++)
        {
            csr_offset_tmp[vertexId] = csr_offset_tmp[vertexId - 1] + static_cast<countl_type>(copResult.csrResult_device.outDegree[vertexId - 1]);
        }
        copResult.csr_offset_CPUOwn = csr_offset_tmp;
        // check
        assert_msg(curEdge == copResult.csr_offset_CPUOwn[copResult.cutVertex + 1], "curEdge = %zu,  copResult.csr_offset_CPUOwn[%zu] = %zu",
                   SCU64(curEdge), SCU64(copResult.cutVertex + 1), SCU64(copResult.csr_offset_CPUOwn[copResult.cutVertex + 1]));
        Msg_info("Build CPU Own csr_offset complete, time = %.2f (ms)", t.get_time_ms());
    }
};

class AvaialibaleMemory
{
  private:
    size_t freeMemory_host;
    size_t totalMemory_host;

    size_t freeMemory_device;
    size_t totalMemory_device;

    int useDeviceId = 0;

  public:
    AvaialibaleMemory(int useDeviceId_ = 0) : freeMemory_host(0), totalMemory_host(0), freeMemory_device(0), totalMemory_device(0)
    {
        useDeviceId = useDeviceId_;
    }

    size_t getTotalMemory_GPU()
    {
        getGPUMemory(freeMemory_device, totalMemory_device, useDeviceId);
        return totalMemory_device;
    }

    size_t getFreeMemory_GPU()
    {
        getGPUMemory(freeMemory_device, totalMemory_device, useDeviceId);
        return freeMemory_device;
    }

    int64_t getAlgFreeMemory_GPU(Algorithm_type algorithm, count_type vertexNum, countl_type edgeNum, size_t reserve = MB(100))
    {
        size_t usedMemory = 0;
        size_t deviceLabe1Length = (vertexNum + 63) / 64;
        getGPUMemory(freeMemory_device, totalMemory_device, useDeviceId);
        switch (algorithm)
        {
        case Algorithm_type::BFS:
            usedMemory += (vertexNum * sizeof(vertex_data_type)); // vertexValue_in
            break;
        case Algorithm_type::SSSP:
            usedMemory += (vertexNum * sizeof(vertex_data_type)); // vertexValue_in
            usedMemory += (deviceLabe1Length * sizeof(uint64_t)); // active_in
            break;
        case Algorithm_type::WCC:
            usedMemory += (vertexNum * sizeof(vertex_data_type));
            usedMemory += (deviceLabe1Length * sizeof(uint64_t)); // active_in
            break;
        case Algorithm_type::PageRank:
            usedMemory += (vertexNum * sizeof(vertex_data_type) * 2); // vertexValue_in + vertexValue_out
            usedMemory += (deviceLabe1Length * sizeof(uint64_t));     // active_in
            break;
        case Algorithm_type::MAX_VALUE:
            assert_msg(false, "You Appoint Error Algorithm_type");
            break;
        }

        usedMemory += reserve;

        return freeMemory_device - usedMemory;
    }

    void getGPUMemory_info(int useDeviceId = 0)
    {
        size_t freeMemory = 0;
        size_t totalMemory = 0;
        CUDA_CHECK(cudaSetDevice(useDeviceId));
        CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));
        Msg_info("The GPU[%d] total memory = %.2lf (GB), used memory = %.2lf (GB), free memory = %.2lf (GB)", useDeviceId,
                 BYTES_TO_GB(SCD(totalMemory)), BYTES_TO_GB(SCD(totalMemory - freeMemory)), BYTES_TO_GB(SCD(freeMemory)));
    }

    void getGPUMemory(size_t &freeMemory, size_t &totalMemory, int useDeviceId = 0)
    {
        CUDA_CHECK(cudaSetDevice(useDeviceId));
        CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));
    }
};
#pragma once

#include "Basic/Balance/taskSteal.hpp"
#include "Basic/basic_include.cuh"
#include "algorithm_device.hpp"
#include "algorithm_host.hpp"
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <mutex>
#include <omp.h>
#include <string>
#include <thread>
#include <unistd.h>

namespace Standalone {

class GraphCG
{
  private:
    Algorithm_type algorithm;
    vertex_id_type root;
    std::string csrWeightFile = "";
    int useDeviceId = 0;
    Monitor *monitor = nullptr;
    static constexpr bool MONITOR = true;
    static constexpr bool DEBUG = true;
    const int _threadNum = omp_get_max_threads();
    static constexpr double INIT_GPU_WORKLOAD = 0.95; // [0.63] best to NATIVE-twitter2010-SSSP
    double INIT_GPU_EDGE_WORKLOAD = 0.85;
    static constexpr int TASK_MEASUREMENT = 0; // 0 FOR VERTEX, 1 FOR EDGES
    static constexpr int OPT_REDUCE = 0;       // 0 FOR OMP_PARALLEL_FOR, 1 FOR TASH_STEAL
    static constexpr bool ENABLE_ZC = 0;

  public:
    Coprocess_result_type &copResult;
    //^ CSR 事实证明这样真的能提高性能,而不是直接调用csrResult取数据
    count_type vertexNum;
    countl_type edgeNum;
    count_type noZeroOutDegreeNum;
    countl_type *csr_offset = nullptr;
    vertex_id_type *csr_dest = nullptr;
    edge_data_type *csr_weight = nullptr;
    countl_type *csr_offset_CPUOwn = nullptr;
    degree_type *outDegree_ownHost = nullptr;   // in host memory
    degree_type *outDegree_ownDevice = nullptr; // in host memory
    degree_type *outDegree = nullptr;           // 为了达到兼容
    degree_type *inDegree = nullptr;            // 为了达到兼容
    const bool isDeviceWhole;
    vertex_id_type cutVertexId;

    countl_type *csr_offset_device = nullptr;
    vertex_id_type *csr_dest_device = nullptr;
    edge_data_type *csr_weight_device = nullptr;
    degree_type *outDegree_device = nullptr;
    degree_type *inDegree_device = nullptr;
    count_type *machine_offset = nullptr;

    //^ Active
    typedef Fixed_Bitset Bitset_type;
    Bitset_type active_in;
    Bitset_type active_out;
    DoubleBuffer<Bitset_type> active;
    size_t *label_cur_device = nullptr;
    size_t *label_next_device = nullptr;
    DoubleBuffer_array<size_t> label_device;
    size_t deviceLabe1Length = 0;

    //^ vertexValue
    vertex_data_type *vertexValue_in = nullptr;
    vertex_data_type *vertexValue_out = nullptr;
    vertex_data_type *vertexValue_result = nullptr;
    vertex_data_type *vertexValue_d2h = nullptr; // used for sync host and device
    vertex_data_type *vertexValue_in_device = nullptr;
    vertex_data_type *vertexValue_out_device = nullptr;
    vertex_data_type *vertexValue_result_device = nullptr;

    //^ stream
    cudaStream_t stream;
    cudaStream_t stream_pr;

    //^ Balance
    Balance::TaskSteal *taskSteal;

    //^ wakeup
    std::mutex GPUCommand_mutex;
    std::condition_variable GPUCommand_cv;
    GPU_state_type GPUCommand = GPU_state_type::MAX_VALUE;
    std::mutex CPUTaskState_mutex;
    int CPUTaskState = 0; // 0: waiting, 1: doing

    //^ co-process
    int ite = 0;
    count_type nBlock = 0;
    count_type nBlock_update = 0;
    int level = 0;
    count_type useWord = 0;
    count_type processVertexNum_GPU = 0;
    count_type sumIte = 0;
    int EXPERIENCE = 0B00;
    int CUR_EXPERIENCE = 0B11;
    vertex_id_type startVertexId_GPU_ZEROCOPY = 0;
    bool enableSubP = false;

  public:
    GraphCG() = delete;
    GraphCG(Coprocess_result_type &copResult_, int useDeviceId_ = 0)
        : copResult(copResult_), vertexNum(0), edgeNum(0), noZeroOutDegreeNum(0), useDeviceId(useDeviceId_), isDeviceWhole(copResult.isDeviceWhole)
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

        //^ host
        vertexNum = copResult.csrResult_host.vertexNum;
        edgeNum = copResult.csrResult_host.edgeNum;
        noZeroOutDegreeNum = copResult.csrResult_host.noZeroOutDegreeNum;

        csr_offset = copResult.csrResult_host.csr_offset;
        csr_dest = copResult.csrResult_host.csr_dest;
        csr_weight = copResult.csrResult_host.csr_weight;
        outDegree_ownHost = copResult.csrResult_host.outDegree;
        outDegree_ownDevice = copResult.csrResult_device.outDegree;
        outDegree = copResult.csrResult_host.outDegree;
        csr_offset_CPUOwn = copResult.csr_offset_CPUOwn;
        cutVertexId = copResult.cutVertex;

        active_in.resize(vertexNum);
        active_out.resize(vertexNum);
        active.setDoubleBuffer(active_in, active_out);

        //^ check GPU memory type
        cudaPointerAttributes attributes;
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, copResult.csrResult_device.csr_offset));
        switch (attributes.type)
        {
        case cudaMemoryTypeUnregistered:
            Msg_info("The device[%d]: [copResult.csrResult_device.csr_offset] is Unregistered", attributes.device);
            assert_msg(false, "[copResult.csrResult_device.csr_offset] must registered");
            break;
        case cudaMemoryTypeHost:
            Msg_info("The device[%d]: [copResult.csrResult_device.csr_offset] is registered to hostPointer", attributes.device);
            assert_msg(false, "[copResult.csrResult_device.csr_offset] is registered to hostPointer, but it is a device pointer");
            break;
        case cudaMemoryTypeDevice:
            Msg_info("The device[%d]: [copResult.csrResult_device.csr_offset] is registered to [%s]", attributes.device,
                     GPU_memory_type_name[SCI32(GPU_memory_type::GPU_MEM)]);
            break;
        case cudaMemoryTypeManaged:
            // we not further check the UVM or ZERO_COPY, which need used [cudaMemRangeGetAttribute]
            // @see https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
            Msg_info("The device[%d]: [copResult.csrResult_device.csr_offset] is registered to [%s]", attributes.device,
                     GPU_memory_type_name[SCI32(GPU_memory_type::UVM)]);
            break;
        }

        deviceLabe1Length = (vertexNum + 63) / 64;

        root = std::numeric_limits<vertex_id_type>::max();

        taskSteal = new Balance::TaskSteal();

        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaStreamCreate(&stream_pr));

        //^ Malloc CPU/GPU Memory
        mallocMemory();
    }
    void setMonitor(Monitor *monitor_) { monitor = monitor_; }
    void setCsrWeightFile(std::string csrWeightFile_) { csrWeightFile = csrWeightFile_; }
    ~GraphCG() { clearGraph(); }

  public:
    double graphProcess(Algorithm_type algorithm_, vertex_id_type root_)
    {
        algorithm = algorithm_;
        root = root_;
        bool isMaxIteCount = false;

        setGPUCommand(GPU_state_type::INIT);
        std::thread agent(&GraphCG::GPU_Process, this);
        getGPUCommand_block(GPU_state_type::WAIT);

        initMemoryForAlg();
        getGPUMemory_info();

        ite = 0;
        double processTime = 0.0;
        count_type activeVertexNum_total = active.in().parallel_popcount(noZeroOutDegreeNum);
        std::pair<size_t, size_t> activeEdgePair;
        bool invokeGPU = false;
        size_t invokeGPU_edgeNum = copResult.invokeGPU_edgeNum; // copResult.invokeGPU_edgeNum;
        double invokeGPU_edgePct = copResult.invokeGPU_edgePct;
        CUR_EXPERIENCE = 0B00;
        EXPERIENCE = 0B00;

        timer iteTime;
        timer singTime;
        timer tmpTime;
        do
        {
            singTime.start(); // Single Ite Time
            ite++;

            tmpTime.start();
            outClear();
            Log_info("The ite[%d]: outClear, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());

            if (CUR_EXPERIENCE >= EXPERIENCE)
            {

                tmpTime.start();
                activeEdgePair = getCurActiveEdgeNum();

                CUR_EXPERIENCE = 0B00;
                if ((activeEdgePair.second >= invokeGPU_edgeNum) && ((SCD(activeEdgePair.second) / SCD(activeEdgePair.first) > invokeGPU_edgePct)))
                {
                    invokeGPU = true;
                    CUR_EXPERIENCE = 0B10;
                    EXPERIENCE |= CUR_EXPERIENCE;
                }
                Log_info(
                    "The ite[%d]: getCurActiveEdgeNum, activeEdges_CPU = %zu, activeEdges_GPU = %zu (%.2lf), invokeGPU = %d, Used time: %.2lf (ms)",
                    ite, activeEdgePair.first, activeEdgePair.second, SCD(activeEdgePair.second) / SCD(activeEdgePair.first), invokeGPU,
                    tmpTime.get_time_ms());
            }

            if (invokeGPU)
            {
                // CUDA_CHECK(cudaMemcpyAsync(vertexValue_in_device, vertexValue_in, (cutVertexId + 1) * sizeof(vertex_data_type),
                //                            cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(vertexValue_in_device, vertexValue_in, (noZeroOutDegreeNum) * sizeof(vertex_data_type),
                                           cudaMemcpyHostToDevice, stream));

                tmpTime.start();
                partitionTask(activeVertexNum_total, activeEdgePair.second);
                Log_info("The ite[%d]: partitionTask, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());

                if constexpr (ENABLE_ZC)
                {
                    if (!isDeviceWhole)
                    {
                        tmpTime.start();
                        enableSubPartition(activeVertexNum_total);
                        Log_info("The ite[%d]: enableSubPartition, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());
                    }
                }

                tmpTime.start();
                CUDA_CHECK(cudaStreamSynchronize(stream));
                Log_info("The ite[%d]: cudaStreamSynchronize, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());

                tmpTime.start();
                assert_msg(GPUCommand == GPU_state_type::WAIT, "The error GPUState: [%s]", GPU_state_type_name[SCI32(GPUCommand)]);
                setGPUCommand(GPU_state_type::RUN);
            }

            //^ CPU
            tmpTime.start();
            Log_info("The ite[%d]: start CPU processing", ite);
            if (!invokeGPU)
            {
                switch (algorithm)
                {
                    // clang-format off
                case Algorithm_type::BFS:
                    taskSteal->taskSteal_edgeLevel(*this, noZeroOutDegreeNum, 0,
                    [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                    {
                        return BFS_HOST_SPACE::BFS(*this, vertex, nbr_start, nbr_end);
                    });                   
                    break;
                case Algorithm_type::SSSP:
                    taskSteal->taskSteal_edgeLevel(*this, noZeroOutDegreeNum, 0,
                    [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                    {
                        return SSSP_HOST_SPACE::SSSP(*this, vertex, nbr_start, nbr_end);
                    });
                    break;
                case Algorithm_type::WCC:
                    taskSteal->taskSteal_edgeLevel(*this, noZeroOutDegreeNum, 0,
                    [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                    {
                        return WCC_HOST_SPACE::WCC(*this, vertex, nbr_start, nbr_end);
                    });
                    break;
                case Algorithm_type::PageRank:
                    //^ compute
                   taskSteal->taskSteal_edgeLevel(*this, noZeroOutDegreeNum, 0,
                    [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                    {
                        return PR_HOST_SPACE::PageRank(*this, vertex, nbr_start, nbr_end);
                    });          
                    //^ update
                    PR_HOST_SPACE::UpdateCoprocess(*this);
                    std::swap(vertexValue_in, vertexValue_out);
                    if (ite > root)
                        isMaxIteCount = true;
                    
                    break;
                case Algorithm_type::MAX_VALUE:
                break;
                    // clang-format on
                }
            }
            else
            {
                if (isDeviceWhole)
                {
                    switch (algorithm)
                    {
                        // clang-format off
                    case Algorithm_type::BFS:
                        taskSteal->taskSteal_edgeLevel_edgeCut(*this, (noZeroOutDegreeNum - processVertexNum_GPU), processVertexNum_GPU,
                        [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                        {
                            return BFS_HOST_SPACE::BFS(*this, vertex, nbr_start, nbr_end);
                        });
                        break;
                    case Algorithm_type::SSSP:
                        taskSteal->taskSteal_edgeLevel_edgeCut(*this, (noZeroOutDegreeNum - processVertexNum_GPU), processVertexNum_GPU,
                        [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                        {
                            return SSSP_HOST_SPACE::SSSP(*this, vertex, nbr_start, nbr_end);
                        });
                        break;
                    case Algorithm_type::WCC:
                    taskSteal->taskSteal_edgeLevel_edgeCut(*this, (noZeroOutDegreeNum - processVertexNum_GPU), processVertexNum_GPU,
                        [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                        {
                            return WCC_HOST_SPACE::WCC(*this, vertex, nbr_start, nbr_end);
                        });
                        break;
                    case Algorithm_type::PageRank:
                        //^ compute
                        taskSteal->taskSteal_edgeLevel_edgeCut(*this, (noZeroOutDegreeNum - processVertexNum_GPU), processVertexNum_GPU,
                        [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                        {
                            return PR_HOST_SPACE::PageRank(*this, vertex, nbr_start, nbr_end);
                        });
                        if (ite > root)
                            isMaxIteCount = true;
                        break;
                    case Algorithm_type::MAX_VALUE:
                    break;
                        // clang-format on
                    }
                }
                else
                {
                    if (enableSubP)
                    {
                        //^ CPU do begin tasks
                        enableSubPartition_CPUTask(startVertexId_GPU_ZEROCOPY, 0, isMaxIteCount);

                        //^ 检查GPU是否有帮忙
                        int cpuState = 0;
                        {
                            std::lock_guard<std::mutex> CPUTaskState_lock(CPUTaskState_mutex);
                            cpuState = CPUTaskState;
                            if (cpuState == 0)
                            {
                                CPUTaskState = 1; // CPU自己来完成
                            }
                        }
                        if (cpuState == 0)
                        {
                            //^ CPU do begin tasks
                            tmpTime.start();
                            enableSubPartition_CPUTask(noZeroOutDegreeNum - startVertexId_GPU_ZEROCOPY, startVertexId_GPU_ZEROCOPY, isMaxIteCount);
                            Log_info("The ite[%d]: CPU Process The ZC, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());
                        }
                    }
                    else
                    {
                        switch (algorithm)
                        {
                            // clang-format off
                        case Algorithm_type::BFS:
                            taskSteal->taskSteal_edgeLevel_edgeCut(*this, (noZeroOutDegreeNum), 0,
                            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                            {
                                return BFS_HOST_SPACE::BFS(*this, vertex, nbr_start, nbr_end);
                            });                   
                            break;
                        case Algorithm_type::SSSP:
                            taskSteal->taskSteal_edgeLevel_edgeCut(*this, (noZeroOutDegreeNum), 0,
                            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                            {
                                return SSSP_HOST_SPACE::SSSP(*this, vertex, nbr_start, nbr_end);
                            });
                            break;
                        case Algorithm_type::WCC:
                            taskSteal->taskSteal_edgeLevel_edgeCut(*this, (noZeroOutDegreeNum), 0,
                            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                            {
                                return WCC_HOST_SPACE::WCC(*this, vertex, nbr_start, nbr_end);
                            });
                            break;
                        case Algorithm_type::PageRank:
                            //^ compute
                            taskSteal->taskSteal_edgeLevel_edgeCut(*this, (noZeroOutDegreeNum), 0,
                            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                            {
                                return PR_HOST_SPACE::PageRank(*this, vertex, nbr_start, nbr_end);
                            });          
                            if (ite > root)
                                isMaxIteCount = true;
                            
                            break;
                        case Algorithm_type::MAX_VALUE:
                        break;
                            // clang-format on
                        }
                    }
                }
            }
            Log_info("The ite[%d]: CPU-Process, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());

            // Wait GPU
            if (invokeGPU)
            {
                tmpTime.start();
                {
                    std::unique_lock<std::mutex> GPUCommand_lock(GPUCommand_mutex);
                    GPUCommand_cv.wait(GPUCommand_lock,
                                       [&]()
                                       {
                        return ((GPUCommand == GPU_state_type::WAIT));
                    });
                }
                assert_msg(GPUCommand == GPU_state_type::WAIT, "GPUState error");
                Log_info("The ite[%d]: Wait GPU Processing, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());

                // Sync CPU And GPU
                tmpTime.start();
                if (algorithm != Algorithm_type::PageRank)
                {
                    if constexpr (OPT_REDUCE == 0)
                    {
                        omp_parallel_for(vertex_id_type vertexId = 0; vertexId <= copResult.cutVertex; vertexId++)
                        {
                            if (vertexValue_in[vertexId] > vertexValue_d2h[vertexId])
                            {
                                vertexValue_in[vertexId] = vertexValue_d2h[vertexId];
                                active.out().set_bit(vertexId);
                            }
                        }
                    }
                    else if constexpr (OPT_REDUCE == 1)
                    {
                        //^ 按照64对齐
                        size_t tasks = copResult.cutVertex + 1;
                        // clang-format off
                        taskSteal->taskSteal_twoStage<size_t, size_t>(tasks,
                           [&](size_t& current, size_t& local_workloads)
                           {
                                size_t end = current + VERTEXWORK_CHUNK;
                                size_t length = VERTEXWORK_CHUNK;
                                if (end >= tasks)
                                    length = tasks - current;
                                for (size_t in = 0; in < length; in++)
                                {
                                    vertex_id_type vertexId = current + in;
                                    if (vertexValue_in[vertexId] > vertexValue_d2h[vertexId])
                                    {
                                        vertexValue_in[vertexId] = vertexValue_d2h[vertexId];
                                        active.out().set_bit(vertexId);
                                    }
                                }
                           }
                        );
                        // clang-format on
                    }
                    else
                    {
                        assert_msg(false, "Error [OPT_REDUCE]");
                    }
                }
                else
                {
                    omp_parallel_for(vertex_id_type vertexId = 0; vertexId <= copResult.cutVertex; vertexId++)
                    {
                        vertexValue_out[vertexId] += vertexValue_d2h[vertexId];
                    }
                    //^ update
                    PR_HOST_SPACE::UpdateCoprocess(*this);
                    std::swap(vertexValue_in, vertexValue_out);
                }
                Log_info("The ite[%d]: Reduce, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());
            }

            level++;
            active.swap();
            invokeGPU = false;
            enableSubP = false;
            {
                std::lock_guard<std::mutex> CPUTaskState_lock(CPUTaskState_mutex);
                CPUTaskState = 0;
            }

            tmpTime.start();
            activeVertexNum_total = active.in().parallel_popcount(noZeroOutDegreeNum);
            Log_info("The ite[%d]: parallel_popcount, activeVertexNum_total = %zu, Used time: %.2lf (ms)", ite, SCU64(activeVertexNum_total),
                     tmpTime.get_time_ms());

            Msg_node("\t[Co-process]: The (%d) iteration, Used time: = %.2lf (ms), activeNum = %zu", ite, singTime.get_time_ms(),
                     SCU64(activeVertexNum_total));

            if (activeVertexNum_total == 0 || isMaxIteCount)
            {
                processTime = iteTime.get_time_ms();
                Msg_node("[Complete]: %s -> iteration: %3d, Used time:: %.6lf (ms)", Algorithm_type_name[SCI32(algorithm)], ite, processTime);
                setGPUCommand(GPU_state_type::EXIT);
                agent.join();
                break;
            }
        }
        while (true);

        return processTime;
    }

  private:
    /* *********************************************************************************************************
     * @description:
     * @return [*]
     * *********************************************************************************************************/
    void GPU_Process()
    {
        int coreId = 0;
        if (threadBindToCore(coreId))
        {
            assert_msg((sched_getcpu() == coreId), "(sched_getcpu() != coreId) -> (%u != %u)", sched_getcpu(), coreId);
            Msg_info("std::thread success bind to core [%u]", coreId);
        }
        else
            Msg_error("std::thread bind to core [%u] failed", coreId);

        timer singleTime;
        timer tmpTime;

        while ((GPUCommand != GPU_state_type::EXIT))
        {
            setGPUCommand(GPU_state_type::WAIT);
            {
                std::unique_lock<std::mutex> GPUCommand_lock(GPUCommand_mutex);
                GPUCommand_cv.wait(GPUCommand_lock,
                                   [&]()
                                   {
                    return ((GPUCommand == GPU_state_type::RUN) || (GPUCommand == GPU_state_type::EXIT));
                });
            } // Wait Run Command

            singleTime.start();
            if (GPUCommand == GPU_state_type::RUN)
            {
                tmpTime.start();
                Log_info("The ite[%d]: begining GPU processing, vertexNumProcess = %zu", ite, SCU64(processVertexNum_GPU));
                switch (algorithm)
                {
                case Algorithm_type::BFS:
                    nBlock = ((processVertexNum_GPU * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                    BFS_DEVICE_SPACE::coalesce_chunk_coProcess_model(*this, nBlock, level, processVertexNum_GPU);
                    break;
                case Algorithm_type::SSSP:
                    nBlock = ((processVertexNum_GPU * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                    SSSP_DEVICE_SPACE::coalesce_chunk_coProcess_model(*this, nBlock, processVertexNum_GPU);
                    break;
                case Algorithm_type::WCC:
                    nBlock = ((processVertexNum_GPU * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                    WCC_DEVICE_SPACE::coalesce_chunk_coProcess_model(*this, nBlock, processVertexNum_GPU);
                    break;
                case Algorithm_type::PageRank:
                    nBlock = ((processVertexNum_GPU * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                    PR_DEVICE_SPACE::coalesce_chunk_coProcess_model(*this, nBlock, processVertexNum_GPU);
                    CUDA_CHECK(cudaMemsetAsync(vertexValue_in_device, 0, sizeof(vertex_data_type) * vertexNum, stream_pr));
                    break;
                case Algorithm_type::MAX_VALUE:
                    break;
                }
                Log_info("The ite[%d]: GPU Processing, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());

                //! help CPU
                if (enableSubP)
                {
                    int cpuState = 0;
                    {
                        std::lock_guard<std::mutex> CPUTaskState_lock(CPUTaskState_mutex);
                        cpuState = CPUTaskState;
                        if (cpuState == 0)
                        {
                            CPUTaskState = 1; // GPU帮其完成
                        }
                    }
                    if (cpuState == 0)
                    {
                        tmpTime.start();
                        if (algorithm != Algorithm_type::BFS)
                        {
                            count_type maxGPUVertexNum_word = ((noZeroOutDegreeNum - startVertexId_GPU_ZEROCOPY) + 63) / 64;
                            assert_msg(noZeroOutDegreeNum >= startVertexId_GPU_ZEROCOPY,
                                       "error: noZeroOutDegreeNum = %zu, startVertexId_GPU_ZEROCOPY = %zu", SCU64(noZeroOutDegreeNum),
                                       SCU64(startVertexId_GPU_ZEROCOPY));
                            assert_msg(startVertexId_GPU_ZEROCOPY % 64 == 0,
                                       "startVertexId_GPU_ZEROCOPY mod 64 != 0, startVertexId_GPU_ZEROCOPY = %zu", SCU64(startVertexId_GPU_ZEROCOPY));
                            Log_info("The ite[%d]: GPU [help CPU] maxGPUVertexNum_word = %zu", ite, SCU64(maxGPUVertexNum_word));
                            CUDA_CHECK(cudaMemcpy(label_cur_device + (startVertexId_GPU_ZEROCOPY / 64),
                                                  active.in().array + (startVertexId_GPU_ZEROCOPY / 64), maxGPUVertexNum_word * sizeof(size_t),
                                                  FLAG_H2D)); // active to GPU
                        }
                        Log_info("The ite[%d]: GPU [help CPU] Label H2D, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());

                        tmpTime.start();
                        count_type help_processVertexNum_CPU = noZeroOutDegreeNum - startVertexId_GPU_ZEROCOPY;
                        switch (algorithm)
                        {
                        case Algorithm_type::BFS:
                            nBlock = ((help_processVertexNum_CPU * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                            BFS_DEVICE_SPACE::coalesce_chunk_coProcess_help_model(*this, nBlock, level, help_processVertexNum_CPU,
                                                                                  startVertexId_GPU_ZEROCOPY);
                            break;
                        case Algorithm_type::SSSP:
                            nBlock = ((help_processVertexNum_CPU * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                            SSSP_DEVICE_SPACE::coalesce_chunk_coProcess_help_model(*this, nBlock, help_processVertexNum_CPU,
                                                                                   startVertexId_GPU_ZEROCOPY);
                            break;
                        case Algorithm_type::WCC:
                            nBlock = ((help_processVertexNum_CPU * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                            // WCC_DEVICE_SPACE::coalesce_chunk_coProcess_model(*this, nBlock, help_processVertexNum_CPU, startVertexId_GPU_ZEROCOPY);
                            break;
                        case Algorithm_type::PageRank:
                            nBlock = ((help_processVertexNum_CPU * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
                            // PR_DEVICE_SPACE::coalesce_chunk_coProcess_model(*this, nBlock, help_processVertexNum_CPU, startVertexId_GPU_ZEROCOPY);
                            CUDA_CHECK(cudaMemsetAsync(vertexValue_in_device, 0, sizeof(vertex_data_type) * vertexNum, stream_pr));
                            break;
                        case Algorithm_type::MAX_VALUE:
                            break;
                        }
                        Log_info("The ite[%d]: GPU Process The ZC, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());
                    }
                }

                tmpTime.start();
                if (enableSubP)
                {
                    CUDA_CHECK(D2H(vertexValue_d2h, vertexValue_result_device, noZeroOutDegreeNum));
                }
                else
                {
                    CUDA_CHECK(D2H(vertexValue_d2h, vertexValue_result_device, copResult.cutVertex + 1));
                }

                Log_info("The ite[%d]: vertexValue D2H, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());

                if (algorithm == Algorithm_type::PageRank)
                {
                    tmpTime.start();
                    CUDA_CHECK(cudaStreamSynchronize(stream_pr));
                    Log_info("The ite[%d]: cudaStreamSynchronize PR, Used time: %.2lf (ms)", ite, tmpTime.get_time_ms());
                    std::swap(vertexValue_in_device, vertexValue_out_device);
                }
            }
            Log_info("The ite[%d]: GPU Total Processing, Used time: %.2lf (ms)", ite, singleTime.get_time_ms());
        }
    } // end of function [GPU_Process()]

  public:
    void measure(std::string file)
    {
        //^ we want used PagaRank to estimate CPU and GPU performance
        double CPU_time = 0.0;
        double GPU_time = 0.0;
        algorithm = Algorithm_type::PageRank;
        initMemoryForAlg();
        if (algorithm == Algorithm_type::PageRank)
        {
            //^ GPU
            timer measure_time;
            nBlock = ((vertexNum * (WARPSIZE / CHUNK_SIZE_EMOGI) + BLOCKSIZE - 1) / BLOCKSIZE);
            PR_DEVICE_SPACE::coalesce_chunk_measure_model(*this, nBlock, copResult.cutVertex + 1);
            GPU_time = measure_time.get_time_ms();

            //^ CPU
            measure_time.start();
            // clang-format off
            taskSteal->taskSteal_threeStage_sink(*this,
            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
            {
                return PR_HOST_SPACE::PageRank(*this, vertex, nbr_start, nbr_end);
            });
            // clang-format on
            CPU_time = measure_time.get_time_ms();

            //^ Ratio
            INIT_GPU_EDGE_WORKLOAD = 1 - (GPU_time / (CPU_time + GPU_time));
            Msg_major("GPU_time = %.2lf (ms), CPU_time = %.2lf (ms), ratio = %.2lf", GPU_time, CPU_time, INIT_GPU_EDGE_WORKLOAD);
            std::ofstream outFile;
            outFile.open(file, std::ios::out);
            assert_msg(outFile.is_open(), "GraphCG-measure [%s] can not open", file.c_str());
            outFile << INIT_GPU_EDGE_WORKLOAD;
            outFile.close();
        }
    }

    void setCGRatio(std::string file)
    {
        std::ifstream inFile;
        inFile.open(file, std::ios_base::in);
        assert_msg(inFile.is_open(), "GraphCG-setCGRatio [%s] can not open", file.c_str());
        std::string line;
        std::getline(inFile, line);
        assert_msg(!line.empty(), "The [CG-ratio.txt] is empty");
        INIT_GPU_EDGE_WORKLOAD = std::stod(line);
        INIT_GPU_EDGE_WORKLOAD = INIT_GPU_EDGE_WORKLOAD * 1.2;
        Msg_info("INIT_GPU_EDGE_WORKLOAD = %.2lf", INIT_GPU_EDGE_WORKLOAD);
        inFile.close();
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

        //^ vertexValue_d2u
        CUDA_CHECK(MALLOC_HOST(&vertexValue_d2h, vertexNum));

        //^ machine_offset
        machine_offset = new count_type[2]; // This is standalone
        machine_offset[0] = 0;
        machine_offset[1] = noZeroOutDegreeNum;

        Msg_info("Malloc CPU/GPU vertexValue_in Memory used time: %.2lf (ms)", malloc_time.get_time_ms());

        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("Malloc CPU/GPU vertexValue_in Memory, used max physical memory: %.2lf (GB)",
                         static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }

        getGPUMemory_info();
    }

    /**********************************************************************************************************
     * @description: Obtain the current usage status of GPUs
     * @return [*]
     **********************************************************************************************************/
    void getGPUMemory_info()
    {
        size_t freeMemory = 0;
        size_t totalMemory = 0;
        CUDA_CHECK(cudaSetDevice(useDeviceId));
        CUDA_CHECK(cudaMemGetInfo(&freeMemory, &totalMemory));
        Msg_info("The GPU[%d] total memory = %.2lf (GB), used memory = %.2lf (GB), free memory = %.2lf (GB)", useDeviceId,
                 BYTES_TO_GB(SCD(totalMemory)), BYTES_TO_GB(SCD(totalMemory - freeMemory)), BYTES_TO_GB(SCD(freeMemory)));
    }

    /**********************************************************************************************************
     * @description:  Check Memory For The Specified  Algorithm
     **********************************************************************************************************/
    void checkMemoryForAlg()
    {
        switch (algorithm)
        {
        case Algorithm_type::BFS:
            break;
        case Algorithm_type::SSSP:
            if (copResult.csrResult_host.csr_weight == nullptr)
            {
                assert_msg(false, "You Ready Run %s, but you [copResult.csrResult_host.csr_weight] is nullptr",
                           Algorithm_type_name[SCI32(algorithm)]);
            }
            if (copResult.csrResult_device.csr_weight == nullptr)
            {
                assert_msg(false, "You Ready Run %s, but you [copResult.csrResult_device.csr_weight] is nullptr",
                           Algorithm_type_name[SCI32(algorithm)]);
            }
            if (label_cur_device == nullptr)
            {
                CUDA_CHECK(MALLOC_DEVICE(&label_cur_device, deviceLabe1Length));
            }
            break;
        case Algorithm_type::WCC:
            if (label_cur_device == nullptr)
            {
                CUDA_CHECK(MALLOC_DEVICE(&label_cur_device, deviceLabe1Length));
            }
            break;
        case Algorithm_type::PageRank:
            if (vertexValue_out == nullptr)
            {
                CUDA_CHECK(MALLOC_HOST(&vertexValue_out, vertexNum));
                CUDA_CHECK(MALLOC_DEVICE(&vertexValue_out_device, vertexNum));
            }
            if (label_cur_device == nullptr)
            {
                CUDA_CHECK(MALLOC_DEVICE(&label_cur_device, deviceLabe1Length));
            }
            break;
        case Algorithm_type::MAX_VALUE:
            assert_msg(false, "You Appoint Error Algorithm_type");
            break;
        }
        return;
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
        case Algorithm_type::SSSP:
            //^ root check
            if (root == std::numeric_limits<vertex_id_type>::max())
                assert_msg_clear(false, "You Ready Run %s, but you do not appoint root", Algorithm_type_name[SCI32(algorithm)]);
            if (root >= noZeroOutDegreeNum)
                assert_msg_clear(false, "You Ready Run %s, but you appoint a invalid root, root = %zu", Algorithm_type_name[SCI32(algorithm)],
                                 static_cast<uint64_t>(root));
            //^ init the vertexValue
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                vertexValue_in[vertexId] = std::numeric_limits<vertex_data_type>::max();
            }
            //^ active
            vertexValue_in[root] = 0;
            CUDA_CHECK(H2D(vertexValue_in_device, vertexValue_in, vertexNum));
            active.in().set_bit(root);
            //^ vertexValue_result
            vertexValue_result = vertexValue_in;               // 不用vertexValue_out的原因是vertexValue_out为nullptr
            vertexValue_result_device = vertexValue_in_device; // vertexValue_out_device -> costList
            break;
        case Algorithm_type::WCC:
            //^ init the vertexValue
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++) { vertexValue_in[vertexId] = vertexId; }
            CUDA_CHECK(H2D(vertexValue_in_device, vertexValue_in, vertexNum));
            //^ active
            active.in().fill_smart();
            active.out().clear_smart();
            //^ vertexValue_result
            vertexValue_result = vertexValue_in;
            vertexValue_result_device = vertexValue_in_device;
            break;
        case Algorithm_type::PageRank:
            //^ maxIte check
            // Msg_warn("You Ready Max Run %s, run iteration : %zu", Algorithm_type_name[SCI32(algorithm)], static_cast<uint64_t>(root));
            //^ init the vertexValue
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                vertexValue_in[vertexId] = 1.0;
                if (outDegree_ownHost[vertexId] != 0)
                    vertexValue_in[vertexId] = 1.0 / outDegree_ownHost[vertexId];
                vertexValue_out[vertexId] = 0.0;
            }
            CUDA_CHECK(H2D(vertexValue_in_device, vertexValue_in, vertexNum));
            CUDA_CHECK(H2D(vertexValue_out_device, vertexValue_out, vertexNum));
            //^ active
            active.in().fill_smart();
            active.out().clear_smart();
            //^ vertexValue_result
            vertexValue_result = vertexValue_in;
            vertexValue_result_device = vertexValue_out_device;
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

    /**********************************************************************************************************
     * @description: Return only wehen GPUState is [gpuCommand_], otherwise it will remain in a blocked state
     * @param [GPU_state_type] gpuCommand_
     * @return [*]
     **********************************************************************************************************/
    void getGPUCommand_block(GPU_state_type gpuCommand_)
    {
        std::unique_lock<std::mutex> GPUCommand_lock(GPUCommand_mutex);
        GPUCommand_cv.wait(GPUCommand_lock,
                           [&]()
                           {
            return (GPUCommand == gpuCommand_);
        });
    }

    /**********************************************************************************************************
     * @description: Return only wehen [isReady_] is true, otherwise it will remain in a blocked state
     * @param [bool] isReady_
     * @return [*]
     **********************************************************************************************************/
    void getGPUCommand_block(bool isReady_)
    {
        std::unique_lock<std::mutex> GPUCommand_lock(GPUCommand_mutex);
        GPUCommand_cv.wait(GPUCommand_lock,
                           [&]()
                           {
            return isReady_;
        });
    }

    /**********************************************************************************************************
     * @description: Set the GPUState to [gpuCommand_]
     * @param [GPU_state_type] gpuCommand_
     * @return [*]
     **********************************************************************************************************/
    void setGPUCommand(GPU_state_type gpuCommand_)
    {
        {
            std::lock_guard<std::mutex> GPUCommand_lock(GPUCommand_mutex);
            GPUCommand = gpuCommand_;
        }
        GPUCommand_cv.notify_one();
    }

    /**********************************************************************************************************
     * @description: Clear the active.out()
     * @return [*]
     **********************************************************************************************************/
    inline void outClear() { active.out().clear_smart(); }

    /**********************************************************************************************************
     * @description:
     * @return [*]
     **********************************************************************************************************/
    std::pair<size_t, size_t> getCurActiveEdgeNum()
    {
        // clang-format off
        std::pair<size_t, size_t> activeEdgePair = taskSteal->taskSteal_twoStage_doubleReturn<size_t, size_t>(
            SCU64(noZeroOutDegreeNum), 
            [&](size_t &current, size_t &local_workloads_first, size_t &local_workloads_second)
            {
                size_t word = active.in().array[WORD_OFFSET(current)];
                while (word != 0)
				{
					if (word & 1)
					{
						local_workloads_first += outDegree_ownHost[current];
                        if((!isDeviceWhole) && (current <= copResult.cutVertex))
                        {
                            local_workloads_second += outDegree_ownDevice[current];
                        }
					}
					current++;
					word = word >> 1;
				}
            }
        );
        // clang-format on
        if (isDeviceWhole)
            activeEdgePair.second = activeEdgePair.first;
        assert_msg(activeEdgePair.first >= activeEdgePair.second, "activeEdgePair error, fisrt = %zu, second = %zu", activeEdgePair.first,
                   activeEdgePair.second);

        return activeEdgePair;
    }

    /**********************************************************************************************************
     * @description: Since Our Reorder Method Can Gather The Most Edges In The Left Front
     *               So We Used Active Vertices To Partition The Graph
     * @return [*]
     **********************************************************************************************************/
    void partitionTask(count_type activeVertexNumTotal_, countl_type activeEdgeNumTotal_GPU_)
    {
        if constexpr (TASK_MEASUREMENT == 0)
        {
            count_type wordId;
            double threshold = INIT_GPU_WORKLOAD * activeVertexNumTotal_;
            count_type wordNum_tmp = (copResult.cutVertex + 1 + 63) / 64;
            assert_msg(wordNum_tmp <= active.in().arrlen, "wordNum_tmp error, wordNum_tmp = %zu", SCU64(wordNum_tmp));
            // Msg_warn("wordNum_tmp = %zu", SCU64(wordNum_tmp));
            count_type sum_ = 0;
            for (wordId = 0; wordId <= wordNum_tmp; wordId++)
            {
                sum_ += __builtin_popcountl(active.in().array[wordId]);
                if (sum_ > threshold)
                {
                    break;
                }
            }
            useWord = ((wordId == 0) ? 1 : wordId); // 兼容小图
            sumIte = sum_;
            assert_msg(sumIte <= activeVertexNumTotal_, "Error: (sumIte > activeVertexNumTotal_), sumIte = %zu, activeVertexNumTotal_ = %zu",
                       SCU64(sumIte), SCU64(activeVertexNumTotal_));

            // Msg_warn("useWord = %zu, threshold = %f, sum_ = %zu, wordId = %zu", SCU64(useWord), threshold, SCU64(sum_), SCU64(wordId));

            if ((useWord * 64) <= copResult.cutVertex)
            {
                processVertexNum_GPU = useWord * 64;
                Log_info("The ite[%d]: partitionTask(VERTEX) wordId = %zu, (PART)vertexNum_to_GPU = %zu", ite, SCU64(useWord),
                         SCU64(processVertexNum_GPU));
            }
            else
            {
                if (isDeviceWhole)
                {
                    processVertexNum_GPU = copResult.cutVertex + 1;
                }
                else
                {
                    processVertexNum_GPU = wordNum_tmp * 64;
                }
                Log_info("The ite[%d]: partitionTask(VERTEX) wordId = %zu, (ALL)vertexNum_to_GPU = %zu", ite, SCU64(useWord),
                         SCU64(processVertexNum_GPU));
            }

            if (isDeviceWhole)
            {
                if (algorithm != Algorithm_type::BFS)
                {
                    CUDA_CHECK(cudaMemcpy(label_cur_device, active.in().array, useWord * sizeof(size_t), FLAG_H2D)); // active to GPU
                }
                memset(active.in().array, 0, sizeof(size_t) * useWord); // clear CPU (CPU 只执行剩余的)
            }
            else
            {
                if (algorithm != Algorithm_type::BFS)
                {
                    count_type maxGPUVertexNum_word = (processVertexNum_GPU + 63) / 64;
                    CUDA_CHECK(cudaMemcpy(label_cur_device, active.in().array, maxGPUVertexNum_word * sizeof(size_t), FLAG_H2D)); // active to GPU
                }
            }
        }
        else if constexpr (TASK_MEASUREMENT == 1)
        {
            double threshold = INIT_GPU_EDGE_WORKLOAD * activeEdgeNumTotal_GPU_;
            int usedThreadNum_tmp = INIT_GPU_EDGE_WORKLOAD / 0.3;
            assert_msg(usedThreadNum_tmp <= ThreadNum, "TASK_MEASUREMENT meet error usedThreadNum_tmp, usedThreadNum_tmp = %d", usedThreadNum_tmp);
            size_t sum_global = 0;
            count_type wordId;
            for (wordId = 0; wordId < active.in().arrlen; wordId += usedThreadNum_tmp)
            {
#pragma omp parallel num_threads(usedThreadNum_tmp) reduction(+ : sum_global)
                {
                    size_t threadId = omp_get_thread_num();
                    size_t word = active.in().array[wordId + threadId];
                    vertex_id_type cur_vertexId = (wordId + threadId) * 64;
                    countl_type sum_local = 0;
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            if (cur_vertexId <= copResult.cutVertex)
                            {
                                sum_local += outDegree_ownDevice[cur_vertexId];
                            }
                        }
                        cur_vertexId++;
                        word = word >> 1;
                    }
                    sum_global += sum_local;
                }
                if (sum_global >= threshold)
                {
                    useWord = (wordId == 0) ? 1 : wordId; // 兼容小图
                    break;
                }
            }
            Log_info("The ite[%d]: partitionTask(EDGE) wordId = %zu, vertexNum_to_GPU = %zu", ite, SCU64(useWord), SCU64(useWord * 64));

            if ((useWord * 64) <= copResult.cutVertex)
            {
                processVertexNum_GPU = useWord * 64;
            }
            else
            {
                processVertexNum_GPU = copResult.cutVertex + 1;
            }

            if (isDeviceWhole)
            {
                if (algorithm != Algorithm_type::BFS)
                {
                    CUDA_CHECK(cudaMemcpy(label_cur_device, active.in().array, useWord * sizeof(size_t), FLAG_H2D)); // active to GPU
                }
                memset(active.in().array, 0, sizeof(size_t) * useWord); // clear CPU (CPU 只执行剩余的)
            }
            else
            {
                if (algorithm != Algorithm_type::BFS)
                {
                    count_type maxGPUVertexNum_word = (processVertexNum_GPU + 63) / 64;
                    CUDA_CHECK(cudaMemcpy(label_cur_device, active.in().array, maxGPUVertexNum_word * sizeof(size_t), FLAG_H2D)); // active to GPU
                }
            }
        }
        else
        {
            assert_msg(false, "Unknown TASK_MEASUREMENT");
        }
    }

    void enableSubPartition(count_type activeVertexNumTotal_)
    {
        static_assert(TASK_MEASUREMENT == 0, "enableSubPartition current only support (TASK_MEASUREMENT == 0)");
        constexpr double TOL = 0.6;
        count_type vertexNum_GPU_process = sumIte;
        count_type vertexNum_CPU_process = activeVertexNumTotal_ - sumIte;
        vertex_id_type subPartition_cutVertexId = 0;
        double INIT_GPU_WORKLOAD_ZEROCOPY = INIT_GPU_WORKLOAD * 1; // ZERO-COPY is slow than GPU_MEM
        double ratio_tmp =
            (SCD(vertexNum_CPU_process) / (1 - INIT_GPU_WORKLOAD_ZEROCOPY)) / (SCD(vertexNum_GPU_process) / (INIT_GPU_WORKLOAD_ZEROCOPY));
        Msg_major("vertexNum_GPU_process = %zu, vertexNum_CPU_process = %zu, ratio_tmp = %.2lf", SCU64(vertexNum_GPU_process),
                  SCU64(vertexNum_CPU_process), ratio_tmp);
        if ((ratio_tmp >= TOL) && (vertexNum_CPU_process > 409600))
        {
            enableSubP = true; //! ------
            int64_t wordId;

            double threshold = INIT_GPU_WORKLOAD_ZEROCOPY * vertexNum_CPU_process;
            count_type sum = 0;
            // if (!isDeviceWhole)
            assert_msg(processVertexNum_GPU % 64 == 0, "processVertexNum_GPU mod 64 need 0, processVertexNum_GPU = %zu", SCU64(processVertexNum_GPU));
            count_type wordStart = (noZeroOutDegreeNum + 63) / 64;
            for (wordId = wordStart - 1; wordId > WORD_OFFSET(processVertexNum_GPU); wordId--)
            {
                sum += __builtin_popcountl(active.in().array[wordId]);
                if (sum > threshold)
                {
                    break;
                }
            }
            startVertexId_GPU_ZEROCOPY = wordId * 64;
            Msg_major("The ite[%d]: total_activeNum = %zu, GPU_activeNum = %zu, CPU_activeNum = %zu", ite, SCU64(activeVertexNumTotal_),
                      SCU64(vertexNum_GPU_process), SCU64(vertexNum_CPU_process));
            Msg_major("The ite[%d]: total_vertexNum = %zu, GPU_vertexNum = %zu, CPUzerocopy_activeNum = %zu", ite, SCU64(noZeroOutDegreeNum),
                      SCU64(processVertexNum_GPU), SCU64(startVertexId_GPU_ZEROCOPY));
        }
    }

    inline void enableSubPartition_CPUTask(size_t totalTaskSize, size_t startTask, bool &isMaxIteCount_)
    {
        switch (algorithm)
        {
            // clang-format off
        case Algorithm_type::BFS:
            taskSteal->taskSteal_edgeLevel_edgeCut(*this, (totalTaskSize), startTask,
            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
            {
                return BFS_HOST_SPACE::BFS(*this, vertex, nbr_start, nbr_end);
            });                   
            break;
        case Algorithm_type::SSSP:
            taskSteal->taskSteal_edgeLevel_edgeCut(*this, (totalTaskSize), startTask,
            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
            {
                return SSSP_HOST_SPACE::SSSP(*this, vertex, nbr_start, nbr_end);
            });
            break;
        case Algorithm_type::WCC:
            taskSteal->taskSteal_edgeLevel_edgeCut(*this, (totalTaskSize), startTask,
            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
            {
                return WCC_HOST_SPACE::WCC(*this, vertex, nbr_start, nbr_end);
            });
            break;
        case Algorithm_type::PageRank:
            //^ compute
            taskSteal->taskSteal_edgeLevel_edgeCut(*this, (totalTaskSize), startTask,
            [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
            {
                return PR_HOST_SPACE::PageRank(*this, vertex, nbr_start, nbr_end);
            });          
            if (ite > root)
                isMaxIteCount_ = true;
            
            break;
        case Algorithm_type::MAX_VALUE:
        break;
            // clang-format on
        }
    }

  public:
    void clearGraph()
    {
        copResult.csrResult_host.clearCSR();

        if (copResult.csrResult_device.csr_offset != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(copResult.csrResult_device.csr_offset));
            copResult.csrResult_device.csr_offset = nullptr;
        }
        if (copResult.csrResult_device.csr_dest != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(copResult.csrResult_device.csr_dest));
            copResult.csrResult_device.csr_dest = nullptr;
        }
        if (copResult.csrResult_device.csr_weight != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(copResult.csrResult_device.csr_weight));
            copResult.csrResult_device.csr_weight = nullptr;
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

        if (vertexValue_d2h != nullptr)
        {
            CUDA_CHECK(FREE_HOST(vertexValue_d2h));
            vertexValue_d2h = nullptr;
        }

        if (vertexValue_out_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(vertexValue_out_device));
            vertexValue_out_device = nullptr;
        }

        if (label_cur_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(label_cur_device));
            label_cur_device = nullptr;
        }

        if (label_next_device != nullptr)
        {
            CUDA_CHECK(FREE_DEVICE(label_next_device));
            label_next_device = nullptr;
        }

        if (machine_offset != nullptr)
        {
            delete[] machine_offset;
            machine_offset = nullptr;
        }

        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("~GraphCG(), used max physical memory: %.2lf (GB)",
                         static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }
        getGPUMemory_info();
    }

}; // end of class  [GraphCG]

} // namespace Standalone
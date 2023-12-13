#pragma once

#include "Basic/Balance/taskSteal.hpp"
#include "Basic/Console/console_V3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Other/IO.hpp"
#include "Basic/Type/data_type.hpp"
#include "Basic/basic_include.cuh"
#include "algorithm_host.hpp"
#include <algorithm>
#include <bits/types/FILE.h>
#include <filesystem>
#include <limits>

namespace Standalone {
class GraphCores
{

  private:
    CSR_Result_type &csrResult;
    Algorithm_type algorithm;
    vertex_id_type root;
    std::string csrWeightFile = "";

    Monitor *monitor = nullptr;
    static constexpr bool MONITOR = true;
    static constexpr bool DEBUG = true;

  public:
    count_type vertexNum;
    countl_type edgeNum;
    countl_type *csr_offset = nullptr;
    vertex_id_type *csr_dest = nullptr;
    edge_data_type *csr_weight = nullptr;
    degree_type *outDegree = nullptr;
    degree_type *inDegree = nullptr;

    //^ Active
    typedef Fixed_Bitset Bitset_type;
    Bitset_type active_in;
    Bitset_type active_out;
    DoubleBuffer<Bitset_type> active;

    // TaskSteal
    Balance::TaskSteal *taskSteal;

    //^ vertexValue
    vertex_data_type *vertexValue_in = nullptr;
    vertex_data_type *vertexValue_out = nullptr;
    vertex_data_type *vertexValue_result = nullptr;

    //^ check result
    count_type *machine_offset = nullptr;

    //^ Sink
    bool sinkFlag = false;
    CSC_Result_type *sinkCSC = nullptr;
    count_type vertexNum_all; // 这是总的(包括sink vertices)
    count_type noZeroOutDegreeNum;

    //^ BFS
    count_type level = 0;

    //>
    GraphCores() = delete;
    GraphCores(CSR_Result_type &csrResult_) : csrResult(csrResult_), vertexNum(0), vertexNum_all(0), edgeNum(0), noZeroOutDegreeNum(0)
    {
        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("Graph(), used max physical memory: %.2lf (GB)", SCD(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }

        // 获得原图数据的CSR
        vertexNum = csrResult.vertexNum;
        edgeNum = csrResult.edgeNum;
        noZeroOutDegreeNum = csrResult.noZeroOutDegreeNum;
        csr_offset = csrResult.csr_offset;
        csr_dest = csrResult.csr_dest;
        csr_weight = csrResult.csr_weight;
        outDegree = csrResult.outDegree;
        inDegree = csrResult.inDegree;

        taskSteal = new Balance::TaskSteal();

        active_in.resize(vertexNum);
        active_out.resize(vertexNum);
        active.setDoubleBuffer(active_in, active_out); // 封装到DoubleBuffer中

        sinkFlag = false;
        vertexNum_all = vertexNum;

        mallocMemory();
    }

    GraphCores(CSR_Result_type &csrResult_, CSC_Result_type &sinkCSC_)
        : csrResult(csrResult_), vertexNum(0), vertexNum_all(0), edgeNum(0), noZeroOutDegreeNum(0)
    {
        // 获得原图数据的CSR
        vertexNum = csrResult.vertexNum;
        vertexNum_all = csrResult.vertexNum + sinkCSC_.vertexNum;
        edgeNum = csrResult.edgeNum;
        noZeroOutDegreeNum = csrResult.noZeroOutDegreeNum;
        assert_msg(vertexNum == noZeroOutDegreeNum, "vertexNum != noZeroOutDegreeNum");
        csr_offset = csrResult.csr_offset;
        csr_dest = csrResult.csr_dest;
        csr_weight = csrResult.csr_weight;
        outDegree = csrResult.outDegree;
        inDegree = csrResult.inDegree;
        Msg_info("vertexNum = %zu, vertexNum_all = %zu", SCU64(vertexNum), SCU64(vertexNum_all));

        taskSteal = new Balance::TaskSteal();

        active_in.resize(vertexNum);
        active_out.resize(vertexNum);
        active.setDoubleBuffer(active_in, active_out); // 封装到DoubleBuffer中

        sinkFlag = true;
        sinkCSC = &sinkCSC_;
        assert_msg((sinkCSC != nullptr), "sinkCSC error");
        Msg_info("sinkCSC->vertexNum = %zu", static_cast<uint64_t>(sinkCSC->vertexNum));

        mallocMemory();
    }
    void setMonitor(Monitor *monitor_) { monitor = monitor_; }
    void setCsrWeightFile(std::string csrWeightFile_) { csrWeightFile = csrWeightFile_; }

    //> Graph Processing
  public:
    double graphProcess(Algorithm_type algorithm_, vertex_id_type root_)
    {
        algorithm = algorithm_;
        root = root_;

        initMemoryForAlg();

        int ite = 0;
        double processTime = 0.0;
        count_type activeNum = 0;
        level = 0;

        timer iteTime;
        timer singTime;
        do
        {
            ite++;
            singTime.start();

            outClear();

            switch (algorithm)
            {
            // clang-format off
            case Algorithm_type::BFS:
                activeNum = 0;
                activeNum = taskSteal->taskSteal_threeStage<GraphCores>(*this,
                [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                {
                    return BFS_HOST_SPACE::BFS(*this, vertex, nbr_start, nbr_end);
                });
                level++;
                break;
            case Algorithm_type::SSSP:
                activeNum = 0;
                activeNum = taskSteal->taskSteal_threeStage<GraphCores>(*this,
                [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                {
                    return SSSP_HOST_SPACE::SSSP(*this, vertex, nbr_start, nbr_end);
                });
                break;
            case Algorithm_type::WCC:
                activeNum = 0;
                activeNum = taskSteal->taskSteal_threeStage<GraphCores>(*this,
                [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                {
                    return WCC_HOST_SPACE::WCC(*this, vertex, nbr_start, nbr_end);
                });
                break;
            case Algorithm_type::PageRank:
                activeNum = 0;
                //^ compute
                taskSteal->taskSteal_threeStage<GraphCores>(*this,
                [&](vertex_id_type vertex, countl_type nbr_start, countl_type nbr_end)
                {
                    return PR_HOST_SPACE::PageRank(*this, vertex, nbr_start, nbr_end);
                });
                //^ update
                activeNum = PR_HOST_SPACE::Update(*this);
                std::swap(vertexValue_in, vertexValue_out);
                break;
            case Algorithm_type::MAX_VALUE:
                break;
                // clang-format on
            }

            Msg_node("\t[Mutil-Cores]: The (%d) iteration, Used time: = %.2lf (ms), activeNum = %zu", ite, singTime.get_time_ms(), SCU64(activeNum));

            if (activeNum == 0)
            {
                processTime = iteTime.get_time_ms();
                Msg_node("[Complete]: %s -> iteration: %3d, Used time:: %.6lf (ms)", Algorithm_type_name[SCI32(algorithm)], ite, processTime);
                break;
            }

            active.swap();
        }
        while (true);

        if (sinkFlag)
            processSink();

        return processTime;
    }

    //> 内部函数
  private:
    /**********************************************************************************************************
     * @description: Malloc The VertexValue_in
     **********************************************************************************************************/
    void mallocMemory()
    {
        vertexValue_in = new vertex_data_type[vertexNum_all];
        machine_offset = new count_type[2];
        machine_offset[0] = 0;
        machine_offset[1] = vertexNum; // vertexNum_all

        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("Graph mallocMemory, used max physical memory: %.2lf (GB)",
                         static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }
    }

    /**********************************************************************************************************
     * @description:  Check Memory For The Specified  Algorithm
     **********************************************************************************************************/
    void checkMemoryForAlg()
    {
        switch (algorithm)
        {
        case Algorithm_type::BFS:
            free_csrWeight();
            free_vertexValueOut();
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
            free_vertexValueOut();
            break;
        case Algorithm_type::WCC:
            free_csrWeight();
            free_vertexValueOut();
            break;
        case Algorithm_type::PageRank:
            free_csrWeight();
            if (vertexValue_out == nullptr)
            {
                vertexValue_out = new vertex_data_type[vertexNum_all];
            }
            break;
        case Algorithm_type::MAX_VALUE:
            assert_msg_clear(false, "You Not Appoint Suitable Algorithm");
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
                assert_msg_clear(false, "You Ready Run %s, but you do not appoint root", Algorithm_type_name[SCI32(Algorithm_type::BFS)]);
            if (root >= vertexNum_all)
                assert_msg_clear(false, "You Ready Run %s, but you appoint a invalid root, root = %zu",
                                 Algorithm_type_name[SCI32(Algorithm_type::BFS)], static_cast<uint64_t>(root));
            // 初始话顶点的值
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum_all; vertexId++)
            {
                vertexValue_in[vertexId] = std::numeric_limits<vertex_data_type>::max();
            }
            // active
            active.in().clear_smart();
            active.out().clear_smart();

            active.in().set_bit(root);
            vertexValue_in[root] = 0;

            // vertexValue_result
            vertexValue_result = vertexValue_in;

            break;

        case Algorithm_type::SSSP:
            // root 检查
            if (root == std::numeric_limits<vertex_id_type>::max())
                assert_msg_clear(false, "You Ready Run %s, but you do not appoint root", Algorithm_type_name[SCI32(Algorithm_type::SSSP)]);
            if (root >= vertexNum_all)
                assert_msg_clear(false, "You Ready Run %s, but you appoint a invalid root, root = %zu",
                                 Algorithm_type_name[SCI32(Algorithm_type::SSSP)], static_cast<uint64_t>(root));
            // 初始话顶点的值
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum_all; vertexId++)
            {
                vertexValue_in[vertexId] = std::numeric_limits<vertex_data_type>::max(); // std::numeric_limits<vertex_data_type>::max();
            }
            // active
            active.in().clear_smart();
            active.out().clear_smart();

            active.in().set_bit(root);
            vertexValue_in[root] = 0;

            // vertexValue_result
            vertexValue_result = vertexValue_in;

            break;

        case Algorithm_type::WCC:
            // 初始话顶点的值
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum_all; vertexId++) { vertexValue_in[vertexId] = vertexId; }
            // active
            active.in().fill_smart();
            active.out().clear_smart();

            // vertexValue_result
            vertexValue_result = vertexValue_in;

            break;

        case Algorithm_type::PageRank:
            // 检查root
            Msg_warn("You Ready Run %s, run iteration : %zu", Algorithm_type_name[SCI32(Algorithm_type::PageRank)], static_cast<uint64_t>(root));
            // 初始话顶点的值
            omp_parallel_for(count_type vertexId = 0; vertexId < vertexNum_all; vertexId++) //! PR中sink点为0.0
            {
                if (vertexId < vertexNum)
                    vertexValue_in[vertexId] = 1.0;
                else
                    vertexValue_in[vertexId] = 0.0;
                if (outDegree[vertexId] != 0)
                    vertexValue_in[vertexId] /= outDegree[vertexId];
                vertexValue_out[vertexId] = 0.0;
            }
            // active
            active.in().fill_smart();
            active.out().clear_smart();

            // vertexValue_result
            vertexValue_result = vertexValue_in;
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

    inline void outClear() { active.out().clear_smart(); }

    void processSink()
    {
        double sinkTime = 0.0;
        // 我们利用最后的结果求解所有的sink点的值
        timer sink_time;
        taskSteal->taskSteal_twoStage<size_t, size_t>(
            static_cast<size_t>(sinkCSC->vertexNum),
            [&](size_t &current, size_t &local_workloads)
            {
            size_t end = current + VERTEXWORK_CHUNK;
            size_t length = VERTEXWORK_CHUNK;
            if (end >= sinkCSC->vertexNum)
                length = sinkCSC->vertexNum - current;

            for (size_t in = 0; in < length; in++)
            {
                vertex_id_type vertexId = current + in;

                countl_type firstNbrIndex = sinkCSC->csc_offset[vertexId];
                countl_type lastNbrIndex = sinkCSC->csc_offset[vertexId + 1];

                for (countl_type nbrId = firstNbrIndex; nbrId < lastNbrIndex; nbrId++)
                {
                    vertex_id_type nbr = sinkCSC->csc_src[nbrId];
                    edge_data_type weight = 0;
                    switch (algorithm)
                    {
                    case Algorithm_type::BFS:
                        if ((vertexValue_in[nbr] + 1) < vertexValue_in[vertexId + noZeroOutDegreeNum])
                        {
                            vertexValue_in[vertexId + noZeroOutDegreeNum] = vertexValue_in[nbr] + 1;
                        }
                        break;
                    case Algorithm_type::SSSP:
                        weight = sinkCSC->csc_weight[nbrId];
                        if ((vertexValue_in[nbr] + weight) < vertexValue_in[vertexId + noZeroOutDegreeNum])
                        {
                            vertexValue_in[vertexId + noZeroOutDegreeNum] = vertexValue_in[nbr] + weight;
                        }
                        break;
                    case Algorithm_type::WCC:
                        if (vertexValue_in[nbr] < vertexValue_in[vertexId + noZeroOutDegreeNum])
                        {
                            vertexValue_in[vertexId + noZeroOutDegreeNum] = vertexValue_in[nbr];
                        }
                        break;
                    case Algorithm_type::PageRank:
                        vertexValue_in[vertexId + noZeroOutDegreeNum] += vertexValue_in[nbr];
                        break;
                    case Algorithm_type::MAX_VALUE:
                        break;
                    }
                }
            }
            },
            VERTEXWORK_CHUNK);
        sinkTime = sink_time.get_time_ms();
        Msg_info("Sink Processing Finish, Used time: %.2f (ms)", sinkTime);
    }

  public:
    void free_csrWeight()
    {
        if (csr_weight != nullptr)
        {
            delete[] csr_weight;
            csr_weight = nullptr;
        }
    }

    void free_vertexValueOut()
    {
        if (vertexValue_out != nullptr)
        {
            delete[] vertexValue_out;
            vertexValue_out = nullptr;
        }
    }

    void clearGraph()
    {
        csrResult.clearCSR();

        if (vertexValue_in != nullptr)
        {
            delete[] vertexValue_in;
            vertexValue_in = nullptr;
        }

        if (vertexValue_out != nullptr)
        {
            delete[] vertexValue_out;
            vertexValue_out = nullptr;
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
                Msg_info("~Graph(), used max physical memory: %.2lf (GB)", static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }
    }

}; // end of class [GraphCores]

} // namespace Standalone
#pragma once

#include "Basic/Console/console_V3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Other/IO.hpp"
#include "Basic/Type/data_type.hpp"
#include "Basic/basic_include.cuh"
#include <bits/types/FILE.h>
#include <filesystem>
#include <limits>

namespace Standalone {

class GraphCore
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

    //^ vertexValue
    vertex_data_type *vertexValue_in = nullptr;
    vertex_data_type *vertexValue_out = nullptr;
    vertex_data_type *vertexValue_result = nullptr;

    //^ check result
    count_type *machine_offset = nullptr;

  public:
    GraphCore() = delete;
    GraphCore(CSR_Result_type &csrResult_) : csrResult(csrResult_), vertexNum(0), edgeNum(0)
    {
        if constexpr (MONITOR)
        {
            if (monitor != nullptr)
            {
                Msg_info("Graph(), used max physical memory: %.2lf (GB)", SCD(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);
            }
        }

        vertexNum = csrResult_.vertexNum;
        edgeNum = csrResult_.edgeNum;
        csr_offset = csrResult_.csr_offset;
        csr_dest = csrResult_.csr_dest;
        csr_weight = csrResult_.csr_weight;
        outDegree = csrResult_.outDegree;
        inDegree = csrResult_.inDegree;

        active_in.resize(vertexNum);
        active_out.resize(vertexNum);
        active.setDoubleBuffer(active_in, active_out); // 封装到DoubleBuffer中

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
        count_type local_activeNum = 0;
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
                local_activeNum = 0;
                active.out().clear();
                for (size_t wordId = 0; wordId < active.in().arrlen; wordId++)
                {
                    vertex_id_type vertexId_current = wordId * 64;
                    size_t word = active.in().array[wordId];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = csr_offset[vertexId_current];
                            countl_type nbr_end = csr_offset[vertexId_current + 1];
                            for (countl_type nbrId = nbr_start; nbrId < nbr_end; nbrId++)
                            {
                                vertex_id_type dest = csr_dest[nbrId];
                                if (vertexValue_in[dest] == std::numeric_limits<vertex_data_type>::max())
                                {
                                    vertexValue_in[dest] = level + 1;
                                    active.out().set_bit_unsync(dest);
                                    local_activeNum += 1;
                                }
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
                level++;
                break;
            case Algorithm_type::SSSP:
                local_activeNum = 0;
                active.out().clear();
                for (size_t wordId = 0; wordId < active.in().arrlen; wordId++)
                {
                    vertex_id_type vertexId_current = wordId * 64;
                    size_t word = active.in().array[wordId];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = csr_offset[vertexId_current];
                            countl_type nbr_end = csr_offset[vertexId_current + 1];
                            for (countl_type nbrId = nbr_start; nbrId < nbr_end; nbrId++)
                            {
                                vertex_id_type dest = csr_dest[nbrId];
                                edge_data_type weight = csr_weight[nbrId];
                                vertex_data_type msg = vertexValue_in[vertexId_current] + weight;
                                if (vertexValue_in[dest] > msg)
                                {
                                    vertexValue_in[dest] = msg;
                                    active.out().set_bit_unsync(dest);
                                    local_activeNum += 1;
                                }
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
                break;
            case Algorithm_type::WCC:
                local_activeNum = 0;
                active.out().clear();
                for (size_t wordId = 0; wordId < active.in().arrlen; wordId++)
                {
                    vertex_id_type vertexId_current = wordId * 64;
                    size_t word = active.in().array[wordId];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = csr_offset[vertexId_current];
                            countl_type nbr_end = csr_offset[vertexId_current + 1];
                            for (countl_type nbrId = nbr_start; nbrId < nbr_end; nbrId++)
                            {
                                vertex_id_type dest = csr_dest[nbrId];
                                if (vertexValue_in[dest] > vertexValue_in[vertexId_current])
                                {
                                    vertexValue_in[dest] = vertexValue_in[vertexId_current];
                                    active.out().set_bit_unsync(dest);
                                    local_activeNum += 1;
                                }
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
                break;
            case Algorithm_type::PageRank:
                local_activeNum = 0;
                active.out().clear();
                //^ compute
                for (size_t wordId = 0; wordId < active.in().arrlen; wordId++)
                {
                    vertex_id_type vertexId_current = wordId * 64;
                    size_t word = active.in().array[wordId];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = csr_offset[vertexId_current];
                            countl_type nbr_end = csr_offset[vertexId_current + 1];
                            for (countl_type nbrId = nbr_start; nbrId < nbr_end; nbrId++)
                            {
                                vertex_id_type dest = csr_dest[nbrId];
                                vertexValue_out[dest] += vertexValue_in[vertexId_current];
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }

                //^ update
                for (vertex_id_type vertexId = 0; vertexId < vertexNum; vertexId++)
                {
                    vertexValue_out[vertexId] = (1.0 - Alpha) / vertexNum + Alpha * vertexValue_out[vertexId];
                    if (outDegree[vertexId] > 0)
                    {
                        vertexValue_out[vertexId] /= outDegree[vertexId];
                    }
                    if ((vertexValue_out[vertexId] - vertexValue_in[vertexId]) > Tolerance)
                    {
                        vertexValue_in[vertexId] = 0.0f;
                        active.out().set_bit_unsync(vertexId);
                        local_activeNum += 1;
                    }
                    else
                    {
                        vertexValue_in[vertexId] = 0.0f;
                    }
                }

                break;
            case Algorithm_type::MAX_VALUE:
                break;
            }

            if (local_activeNum == 0)
            {
                processTime = iteTime.get_time_ms();
                break;
            }

            inOutSwap();
        }
        while (true);

        return processTime;
    }

    void saveResultToFile(std::string resultFile)
    {
        if (std::filesystem::exists(resultFile.c_str()))
        {
            Msg_warn("The Result File [%s] Has Existed, Skip Save Current Result", resultFile.c_str());
        }
        else
        {
            std::filesystem::path filePath = resultFile;
            std::filesystem::path folderPath = filePath.parent_path(); // get floder path
            createFloder_recursive(folderPath);
            save_binArray(resultFile, vertexValue_result, SCU64(vertexNum));
        }
    }

    //> 内部函数
  private:
    /**********************************************************************************************************
     * @description: Malloc The VertexValue_in
     **********************************************************************************************************/
    void mallocMemory()
    {
        vertexValue_in = new vertex_data_type[vertexNum];
        machine_offset = new count_type[2];
        machine_offset[0] = 0;
        machine_offset[1] = vertexNum;

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
                vertexValue_out = new vertex_data_type[vertexNum];
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
            if (root >= vertexNum)
                assert_msg_clear(false, "You Ready Run %s, but you appoint a invalid root, root = %zu",
                                 Algorithm_type_name[SCI32(Algorithm_type::BFS)], static_cast<uint64_t>(root));
            // 初始话顶点的值
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                vertexValue_in[vertexId] = std::numeric_limits<vertex_data_type>::max();
            }
            // active
            active.in().clear();
            active.out().clear();
            active.in().set_bit(root);
            vertexValue_in[root] = 0;
            // vertexValue_result
            vertexValue_result = vertexValue_in;
            break;

        case Algorithm_type::SSSP:
            // root 检查
            if (root == std::numeric_limits<vertex_id_type>::max())
                assert_msg_clear(false, "You Ready Run %s, but you do not appoint root", Algorithm_type_name[SCI32(Algorithm_type::SSSP)]);
            if (root >= vertexNum)
                assert_msg_clear(false, "You Ready Run %s, but you appoint a invalid root, root = %zu",
                                 Algorithm_type_name[SCI32(Algorithm_type::SSSP)], static_cast<uint64_t>(root));
            // 初始话顶点的值
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                vertexValue_in[vertexId] = std::numeric_limits<vertex_data_type>::max(); // std::numeric_limits<vertex_data_type>::max();
            }
            // active
            active.in().clear();
            active.out().clear();
            active.in().set_bit(root);
            vertexValue_in[root] = 0;
            // vertexValue_result
            vertexValue_result = vertexValue_in;
            break;

        case Algorithm_type::WCC:
            // 初始话顶点的值
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                vertexValue_in[vertexId] = vertexId;
            }
            // active
            active_in.fill();
            active.out().clear();
            vertexValue_result = vertexValue_in;

            break;

        case Algorithm_type::PageRank:
            // 检查root
            Msg_warn("You Ready Run %s, run iteration : %zu", Algorithm_type_name[SCI32(Algorithm_type::PageRank)], static_cast<uint64_t>(root));
            // 初始话顶点的值
            for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
            {
                vertexValue_in[vertexId] = 1.0;
                if (outDegree[vertexId] != 0)
                    vertexValue_in[vertexId] = 1.0 / outDegree[vertexId];
                vertexValue_out[vertexId] = 0.0;
            }
            // active
            active.in().fill();
            active.out().clear();

            // vertexValue_result
            vertexValue_result = vertexValue_out;
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

    void inOutSwap()
    {
        switch (algorithm)
        {
        case Algorithm_type::BFS:
            active.swap();
            break;
        case Algorithm_type::SSSP:
            active.swap();
            break;
        case Algorithm_type::WCC:
            active.swap();
            break;
        case Algorithm_type::PageRank:
            active.swap();
            std::swap(vertexValue_in, vertexValue_out);
            break;
        case Algorithm_type::MAX_VALUE:
            assert_msg(false, "You Not Appoint Suitable Algorithm");
            break;
        }
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

}; // end of class [GraphCore]
} // namespace Standalone
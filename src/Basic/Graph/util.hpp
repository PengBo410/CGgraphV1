#pragma once

#include "Basic/Console/console_V3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Other/IO.hpp"
#include "Basic/Type/data_type.hpp"
#include "basic_struct.hpp"

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <numaif.h> // move_page
#include <string>
#include <type_traits>

static inline std::string get_BaseGraphFile_path() { return "/data/webgraph/bin/"; }
static inline std::string get_BaseCheckFile_path() { return "/data/webgraph/checkResult/"; }

std::string getResultFilePathV2(CheckInfo_type &checkInfo)
{
    std::string rightFile = "";
    std::string algorithmName = Algorithm_type_name[SCI32(checkInfo.algorithm)];

    switch (checkInfo.algorithm)
    {
    case Algorithm_type::BFS:
        rightFile = get_BaseCheckFile_path() + algorithmName + "/" + checkInfo.graphName + "_" + algorithmName + "_" +
                    std::to_string(checkInfo.root) + ".bin";
        if (!std::filesystem::exists(rightFile))
        {
            Msg_error("When Check Result, Can Not Find Compared File[%s], Skip Result Check...", rightFile.c_str());
            return "";
        }
        return rightFile;
        break;
    case Algorithm_type::SSSP:
        rightFile = get_BaseCheckFile_path() + algorithmName + "/" + checkInfo.graphName + "_" + algorithmName + "_" +
                    std::to_string(checkInfo.root) + ".bin";
        if (!std::filesystem::exists(rightFile))
        {
            Msg_error("When Check Result, Can Not Find Compared File[%s], Skip Result Check...", rightFile.c_str());
            return "";
        }
        return rightFile;
        break;
    case Algorithm_type::WCC:
        rightFile = get_BaseCheckFile_path() + algorithmName + "/" + checkInfo.graphName + "_" + algorithmName + ".bin";
        if (!std::filesystem::exists(rightFile))
        {
            Msg_error("When Check Result, Can Not Find Compared File[%s], Skip Result Check...", rightFile.c_str());
            return "";
        }
        return rightFile;
        break;
    case Algorithm_type::PageRank:
        rightFile = get_BaseCheckFile_path() + algorithmName + "/" + checkInfo.graphName + "_" + algorithmName + ".bin";
        if (!std::filesystem::exists(rightFile))
        {
            Msg_error("When Check Result, Can Not Find Compared File[%s], Skip Result Check...", rightFile.c_str());
            return "";
        }
        return rightFile;
        break;
    case Algorithm_type::MAX_VALUE:
        Msg_error("When Check Result, Get invalid Algorithm, Skip Result Check...");
        return "";
        break;
    }
    return "";
}

template <typename T>
void checkBinResult(std::string resultFilePath, T *current_array, size_t length, size_t checkLength, OrderMethod_type OrderMethod_type, T *old2new)
{
    T *resultArray = load_binFile<T>(resultFilePath, length);

    if (OrderMethod_type == OrderMethod_type::NATIVE)
    {
        omp_parallel_for(count_type i = 0; i < checkLength; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %u, resultArray[%u] = %u", i, current_array[i], i,
                           resultArray[i]);
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %f, resultArray[%u] = %f", i, current_array[i], i,
                           resultArray[i]);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %lf, resultArray[%u] = %lf", i, current_array[i],
                           i, resultArray[i]);
            }
            else
            {
                assert_msg(false, "Function [checkBinResult] Meet Unknown Data_type");
            }
        }
    }
    else
    {
        omp_parallel_for(count_type i = 0; i < checkLength; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                // if(old2new[i] < 4198894)
                assert_msg((current_array[old2new[i]] == resultArray[i]), "检查结果出错:current_array[%u] = %u, resultArray[%u] = %u", old2new[i],
                           current_array[old2new[i]], i, resultArray[i]);
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                assert_msg((current_array[old2new[i]] == resultArray[i]), "检查结果出错:current_array[%u] = %f, resultArray[%u] = %f", old2new[i],
                           current_array[old2new[i]], i, resultArray[i]);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                assert_msg((current_array[old2new[i]] == resultArray[i]), "检查结果出错:current_array[%u] = %lf, resultArray[%u] = %lf", old2new[i],
                           current_array[old2new[i]], i, resultArray[i]);
            }
            else
            {
                assert_msg(false, "Function [checkBinResult] Meet Unknown Data_type");
            }
        }
    }
    if (length == checkLength)
    {
        Msg_finish("All The Algorithm Result Finished The Check !");
    }
    else
    {
        Msg_finish("All The Algorithm Result Finished The Check Expect Sink Vertices !");
    }
}

/*********************************************************************************************************************
 * 检查算法结果的正确性
 * V2.0的变动主要是新增检查的顶点范围，主要服务于Sink Vertices
 *
 * [resultFilePath] 正确的结果所在的路径
 * [current_array]  被检查的结果，也就是当前算法计算出的结果
 * [vertexNum]      总的定点数，包括sink vertices
 * [checkVertexNum] 要执行检查的顶点个数，从vertexId = 0 开始
 * [OrderMethod_type]    排序的方法
 * [old2new]        新旧顶点映射
 *********************************************************************************************************************/
template <typename T>
void checkBinResultV2(std::string resultFilePath, T *current_array, size_t vertexNum, size_t checkVertexNum, OrderMethod_type OrderMethod_type,
                      T *old2new)
{
    T *resultArray = load_binFile<T>(resultFilePath, vertexNum);

    if (OrderMethod_type == OrderMethod_type::NATIVE)
    {
        omp_parallel_for(count_type i = 0; i < checkVertexNum; i++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %u, resultArray[%u] = %u", i, current_array[i], i,
                           resultArray[i]);
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %f, resultArray[%u] = %f", i, current_array[i], i,
                           resultArray[i]);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%u] = %lf, resultArray[%u] = %lf", i, current_array[i],
                           i, resultArray[i]);
            }
            else
            {
                assert_msg(false, "Function [checkBinResult] Meet Unknown Data_type");
            }
        }
    }
    else
    {
        count_type checkVertexNumCount = 0;
#pragma omp parallel for reduction(+ : checkVertexNumCount)
        for (vertex_id_type oldVertexId = 0; oldVertexId < vertexNum; oldVertexId++)
        {
            if constexpr (std::is_same_v<T, uint32_t>)
            {
                vertex_id_type newVertexId = old2new[oldVertexId];
                if (newVertexId < checkVertexNum)
                {
                    assert_msg((current_array[newVertexId] == resultArray[oldVertexId]), "检查结果出错:current_array[%u] = %u, resultArray[%u] = %u",
                               newVertexId, current_array[newVertexId], oldVertexId, resultArray[oldVertexId]);
                    checkVertexNumCount++;
                }
            }
            // todo float和double可以只检查前几位
            else if constexpr (std::is_same_v<T, float>)
            {
                vertex_id_type newVertexId = old2new[oldVertexId];
                if (newVertexId < checkVertexNum)
                {
                    assert_msg((current_array[newVertexId] == resultArray[oldVertexId]), "检查结果出错:current_array[%u] = %u, resultArray[%u] = %u",
                               newVertexId, current_array[newVertexId], oldVertexId, resultArray[oldVertexId]);
                }
                checkVertexNumCount++;
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                vertex_id_type newVertexId = old2new[oldVertexId];
                if (newVertexId < checkVertexNum)
                {
                    assert_msg((current_array[newVertexId] == resultArray[oldVertexId]), "检查结果出错:current_array[%u] = %u, resultArray[%u] = %u",
                               newVertexId, current_array[newVertexId], oldVertexId, resultArray[oldVertexId]);
                }
                checkVertexNumCount++;
            }
            else
            {
                assert_msg(false, "Function [checkBinResult] Meet Unknown Data_type");
            }
        }
        assert_msg(checkVertexNumCount == checkVertexNum, "checkVertexNumCount fail, checkVertexNumCount = %zu, checkVertexNum = %zu",
                   SC(checkVertexNumCount), SC(checkVertexNum));

        if (vertexNum == checkVertexNum)
        {
            Msg_finish("The Algorithm Finished The Result Check !");
        }
        else
        {
            Msg_finish("The Algorithm Finished The Result Check Expect Sink Vertices !");
        }
    }
}

/*********************************************************************************************************************
 * 检查算法结果的正确性
 * V3.0的变动主要是新增检查的顶点范围，主要服务于Sink Vertices 和 Distributed
 * V3.1的变动新增<vertex_data_type> 为float或double时, 支持近似匹配, 近似度定义为: CHECK_Tolerance
 *
 * [resultFilePath] 正确的结果所在的路径
 * [algorithm]      要检查的算法
 * [current_array]  被检查的结果，也就是当前算法计算出的结果
 * [vertexNum]      总的定点数，包括sink vertices
 * [checkVertexNum] 要执行检查的顶点个数，从vertexId = 0 开始
 * [OrderMethod_type]    排序的方法
 * [old2new]        新旧顶点映射
 *********************************************************************************************************************/
template <typename T>
void checkBinResultV3(std::string resultFilePath, T *current_array, size_t vertexNum, size_t checkVertexStart, size_t checkVertexEnd,
                      OrderMethod_type OrderMethod_type, vertex_id_type *old2new)
{
    T *resultArray = load_binFile<T>(resultFilePath, vertexNum);

    if (OrderMethod_type == OrderMethod_type::NATIVE)
    {
        omp_parallel_for(count_type i = checkVertexStart; i < checkVertexEnd; i++)
        {
            //^ 精确的匹配
            if constexpr (std::is_integral_v<T>)
            {
                if (current_array[i] == std::numeric_limits<T>::max()) // 兼容旧版本
                    current_array[i] = VertexValue_MAX;
                assert_msg((current_array[i] == resultArray[i]), "检查结果出错: current_array[%zu] = %zu, resultArray[%zu] = %zu", SCU64(i),
                           SCU64(current_array[i]), SCU64(i), SCU64(resultArray[i]));
            }
            //^ 近似的匹配
            else if constexpr (std::is_floating_point_v<T>)
            {
                assert_msg(std::abs(current_array[i] - resultArray[i]) <= CHECK_Tolerance,
                           "检查结果出错: current_array[%zu] = %lf, resultArray[%zu] = %lf, |%lf| > %lf", SCU64(i), SCD(current_array[i]), SCU64(i),
                           SCD(resultArray[i]), SCD(std::abs(current_array[i] - resultArray[i])), SCD(CHECK_Tolerance));
            }
            else
            {
                assert_msg(false, "You Defined An Error <vertex_data_type>");
            }
        }

        if (vertexNum == (checkVertexEnd - checkVertexStart))
        {
            Msg_finish("Algorithm Finsih The All Vertex Result Check");
        }
        else
        {
            Msg_finish("Algorithm Finsih The Vertex [%zu, %zu) Result Check", static_cast<uint64_t>(checkVertexStart),
                       static_cast<uint64_t>(checkVertexEnd));
        }
    }
    else
    {
        // Msg_warn("checkBinResultV3 current only support [OrderMethod_type::NATIVE], So we skip check ...");
        count_type checkVertexNumCount = 0;
#pragma omp parallel for reduction(+ : checkVertexNumCount)
        for (vertex_id_type oldVertexId = checkVertexStart; oldVertexId < checkVertexEnd; oldVertexId++)
        {
            //^ 精确的匹配
            if constexpr (std::is_integral_v<T>)
            {
                vertex_id_type newVertexId = old2new[oldVertexId];
                if ((newVertexId >= checkVertexStart) && (newVertexId < checkVertexEnd))
                {
                    if (current_array[newVertexId] == std::numeric_limits<T>::max()) // 兼容旧版本
                        current_array[newVertexId] = VertexValue_MAX;
                    assert_msg((current_array[newVertexId] == resultArray[oldVertexId]),
                               "检查结果出错:current_array[%zu] = %zu, resultArray[%zu] = %zu", SCU64(newVertexId), SCU64(current_array[newVertexId]),
                               SCU64(oldVertexId), SCU64(resultArray[oldVertexId]));
                    checkVertexNumCount++;
                }
            }
            //^ 近似的匹配
            else if constexpr (std::is_floating_point_v<T>)
            {
                vertex_id_type newVertexId = old2new[oldVertexId];
                if ((newVertexId >= checkVertexStart) && (newVertexId < checkVertexEnd))
                {
                    assert_msg(std::abs(current_array[newVertexId] - resultArray[oldVertexId]) <= CHECK_Tolerance,
                               "检查结果出错: current_array[%zu] = %lf, resultArray[%zu] = %lf, |%lf| > %lf", SCU64(newVertexId),
                               SCD(current_array[newVertexId]), SCU64(oldVertexId), SCD(resultArray[oldVertexId]),
                               SCD(std::abs(current_array[newVertexId] - resultArray[oldVertexId])), SCD(CHECK_Tolerance));

                    // assert_msg((current_array[newVertexId] == resultArray[oldVertexId]), "检查结果出错:current_array[%u] = %u, resultArray[%u] =
                    // %u",
                    //            newVertexId, current_array[newVertexId], oldVertexId, resultArray[oldVertexId]);
                }
                checkVertexNumCount++;
            }

            else
            {
                assert_msg(false, "You Defined An Error <vertex_data_type>");
            }
        }
        // assert_msg(checkVertexNumCount == (checkVertexEnd - checkVertexStart),
        //            "checkVertexNumCount fail, checkVertexNumCount = %zu, checkVertexNum = %zu", SC(checkVertexNumCount),
        //            SC(checkVertexEnd - checkVertexStart));

        if (vertexNum == (checkVertexEnd - checkVertexStart))
        {
            Msg_finish("Algorithm Finsih The All Vertex Result Check");
        }
        else
        {
            Msg_finish("Algorithm Finsih The Vertex [%zu, %zu) Result Check", static_cast<uint64_t>(checkVertexStart),
                       static_cast<uint64_t>(checkVertexEnd));
        }
    }
}

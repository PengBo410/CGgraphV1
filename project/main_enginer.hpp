#pragma once

#include "Basic/basic_include.cuh"

#include "getGraphCSR.hpp"
#include "gflags/gflags_declare.h"
#include "graph_coProcess.hpp"
#include "graph_core.hpp"
#include "graph_cores.hpp"
#include "graph_device.hpp"
#include "help.hpp"
#include "subgraph_extraction.hpp"

template <typename GraphEnginer>
void profile(GraphEnginer *graphEnginer, std::string graphName, OrderMethod_type orderMethod, vertex_id_type *old2new, count_type vertices,
             Algorithm_type algorithm, int64_t root, int runs, bool rootTranfer, bool logResult, bool isHand);

DECLARE_string(graphName);
DECLARE_int32(orderMethod);
DECLARE_int64(root);
DECLARE_int32(algorithm);
DECLARE_int32(computeEngine);
DECLARE_int32(gpuMemory);
DECLARE_int32(gpuImple);
DECLARE_int32(runs);
// DECLARE_bool(has_csrFile);
// DECLARE_bool(logResult);
DECLARE_int32(useDeviceId);

void main_enginer()
{
    Monitor *monitor = nullptr;
    // monitor = new Monitor();
    if (monitor != nullptr)
        Msg_info("Program begin, max physical memory = %.2lf (GB)", static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);

    std::string graphName = FLAGS_graphName;
    OrderMethod_type orderMethod = static_cast<OrderMethod_type>(FLAGS_orderMethod);
    int64_t root = FLAGS_root;
    Algorithm_type algorithm = static_cast<Algorithm_type>(FLAGS_algorithm);
    ComputeEngine_type computeEngine = static_cast<ComputeEngine_type>(FLAGS_computeEngine);
    GPU_memory_type gpuMemory = static_cast<GPU_memory_type>(FLAGS_gpuMemory);
    GPU_implement_type gpuImple = static_cast<GPU_implement_type>(FLAGS_gpuImple);
    int runs = FLAGS_runs;
    bool has_csrFile = true;
    bool logResult = true;
    int useDeviceId = FLAGS_useDeviceId;

    bool isDataEngineRun = false;
    DataEngine_type dataEngine = DataEngine_type::D2U;
    bool rootTranfer = true;
    bool sink_check = true;
    bool toyExample = false;

    BasicAssert::checkPlatformAndThread();
    BasicAssert::checkAlgorithm(algorithm);
    BasicAssert::checkEnemType(orderMethod, algorithm, computeEngine, gpuMemory, gpuImple);

    // Real-word graph or toy garph
    if (graphName.length() >= 3 && graphName.substr(0, 3).compare("toy") == 0)
        toyExample = true;
    else
        toyExample = false;

    CSR_Result_type csrResult;
    vertex_id_type *old2new;
    if (!toyExample)
    {
        old2new = getGraphCSR(csrResult, graphName, orderMethod, has_csrFile, true, true, false, CSR, sink_check);
    }
    else
    {
        csrResult = getToyGraph(graphName, orderMethod);
    }

    if (monitor != nullptr)
        Msg_info("After build CSR, max physical memory = %.2lf (GB)", static_cast<double>(monitor->getCur_maxPhysicalMemory_KB()) / 1024 / 1024);

    //>  ComputeEngine_type::SINGLE_CORE
    if ((computeEngine == ComputeEngine_type::SINGLE_CORE) && (!isDataEngineRun))
    {
        StaticAssertClass<Standalone::GraphCore>::CHECK();
        Standalone::GraphCore *graph = nullptr;
        graph = new Standalone::GraphCore(csrResult);
        graph->setMonitor(monitor);
        profile<Standalone::GraphCore>(graph, graphName, orderMethod, old2new, csrResult.vertexNum, algorithm, root, runs, rootTranfer, logResult,
                                       false);
    }

    //>  ComputeEngine_type::MULTI_CORE
    else if ((computeEngine == ComputeEngine_type::MULTI_CORE) && (!isDataEngineRun))
    {
        StaticAssertClass<Standalone::GraphCores>::CHECK();
        Standalone::GraphCores *graph = nullptr;
        graph = new Standalone::GraphCores(csrResult);
        graph->setMonitor(monitor);
        profile<Standalone::GraphCores>(graph, graphName, orderMethod, old2new, csrResult.vertexNum, algorithm, root, runs, rootTranfer, logResult,
                                        false);
    }

    //> ComputeEngine_type::COOPERATION
    else if ((computeEngine == ComputeEngine_type::COOPERATION) && (!isDataEngineRun))
    {
        AvaialibaleMemory freeGPUMemory(useDeviceId);
        int64_t freeMem = freeGPUMemory.getAlgFreeMemory_GPU(algorithm, csrResult.vertexNum, csrResult.edgeNum);
        assert_msg(freeMem > 0, "Too Little Available GPU Memory");
        freeGPUMemory.getGPUMemory_info();

        SubgraphExtraction subgraphExtraction(csrResult, freeMem, algorithm, gpuMemory, useDeviceId);
        subgraphExtraction.setMonitor(monitor);
        csrResult.clearCSR();
        Coprocess_result_type copResult = subgraphExtraction.getCoProcessResult();

        std::filesystem::path performancFile = "CG-" + graphName + "-" + OrderMethod_type_name[SCI32(orderMethod)] + ".ratio";
        bool isExist = std::filesystem::exists(performancFile);
        if (!isExist)
        {
            StaticAssertClass<Standalone::GraphCG>::CHECK();
            Standalone::GraphCG *graph_ratio = nullptr;
            graph_ratio = new Standalone::GraphCG(copResult, useDeviceId);
            graph_ratio->measure(performancFile.string());
        }

        StaticAssertClass<Standalone::GraphCG>::CHECK();
        Standalone::GraphCG *graph = nullptr;
        graph = new Standalone::GraphCG(copResult, useDeviceId);
        graph->setMonitor(monitor);
        graph->setCGRatio(performancFile.string());
        profile<Standalone::GraphCG>(graph, graphName, orderMethod, old2new, csrResult.vertexNum, algorithm, root, runs, rootTranfer, logResult,
                                     false);
    }

    //> ComputeEngine_type::SINGLE_GPU
    else if ((computeEngine == ComputeEngine_type::SINGLE_GPU) && (!isDataEngineRun))
    {
        StaticAssertClass<Standalone::GraphDevice>::CHECK();
        Standalone::GraphDevice *graph = nullptr;
        graph = new Standalone::GraphDevice(csrResult, gpuMemory, gpuImple, useDeviceId);
        graph->setMonitor(monitor);
        profile<Standalone::GraphDevice>(graph, graphName, orderMethod, old2new, csrResult.vertexNum, algorithm, root, runs, rootTranfer, logResult,
                                         false);
    }
}

/****************************************************************************************************************
 * 获取运行结果
 *
 * [graphEnginer] 图引擎
 * [graphName]    图名称
 * [orderMethod]  排序方法
 * [old2new]      顶点Id的映射
 * [vertices]     当我们传入sinkCSR后，就无法获取到总的vettexNum,因此需要单独传递总的值
 * [algorithm]    运行的算法
 * [root]         算法的附加信息
 * [runs]         运行次数
 * [rootTranfer]  是否对应排序前后的顶点Id
 * [logResult]    检测结果
 * [isHand]       是否手动输入算法的附加信息 (分布式要注意输入信息的广播)
 ****************************************************************************************************************/
template <typename GraphEnginer>
void profile(GraphEnginer *graphEnginer, std::string graphName, OrderMethod_type orderMethod, vertex_id_type *old2new, count_type vertices,
             Algorithm_type algorithm, int64_t root, int runs, bool rootTranfer, bool logResult, bool isHand)
{
    std::vector<double> usingTime_vec;
    double usingTime = 0.0;
    usingTime_vec.resize(runs);

    do
    {
        if (isHand)
        {
            Msg_info("Input Algorithm Addition Info, e.g. [Root] for BFS/SSSP, [Ite] for PR, [-1] exit: ");
            std::cin >> root;
            if (root == -1)
                break;
            if (root < 0 || root >= graphEnginer->vertexNum)
            {
                Msg_info("Invalid Input, If You Input [Root], It Locate [0, %zu]", static_cast<uint64_t>(graphEnginer->vertexNum));
                continue;
            }
        }

        vertex_id_type switch_root = root;
        if (orderMethod != OrderMethod_type::NATIVE && rootTranfer)
        {
            Msg_info("old-root: %ld, new-root: %u", root, old2new[root]);
            switch_root = old2new[root];
        }
        else
            Msg_info("root no switch: %ld", root);

        for (count_type runId = 0; runId < runs; runId++)
        {
            usingTime_vec[runId] = graphEnginer->graphProcess(algorithm, switch_root);
        }

        sort(usingTime_vec.begin(), usingTime_vec.end());
        if (runs > 2)
        {
            for (count_type runId = 1; runId < runs - 1; runId++)
            {
                usingTime += usingTime_vec[runId];
            }
            Msg_node("[Total Run Time]: %f (ms)", usingTime / (runs - 2));
            usingTime = 0;
        }
        else
        {
            for (count_type runId = 0; runId < runs; runId++)
            {
                usingTime += usingTime_vec[runId];
            }
            Msg_node("[Total Run Time]: %.2f (ms)", usingTime / (runs));
            usingTime = 0;
        }
    }
    while (isHand);
}
#pragma once

#include "Basic/Console/console_V3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/basic_include.cuh"
#include <limits>
#include <utility> // std::forward

template <typename F> class StaticAssertClass
{

  public:
    static constexpr void CHECK() {}

    // 检查类是否包含指定的graphProcess()函数
    template <typename... Args>
    static constexpr auto checkFunction(F *obj, Args &&...args) -> decltype(obj->graphProcess(std::forward<Args>(args)...), bool())
    {
        return true;
    }

    template <typename... Args> static constexpr bool checkFunction(...) { return false; }

    // 检查类是否包含指定的变量
    static constexpr auto checkVariable_vertexNum(F *obj) -> decltype(obj->vertexNum, bool()) { return true; }
    static constexpr auto checkVariable_edgeNum(F *obj) -> decltype(obj->edgeNum, bool()) { return true; }
    static constexpr auto checkVariable_csr_offset(F *obj) -> decltype(obj->csr_offset, bool()) { return true; }
    static constexpr auto checkVariable_csr_dest(F *obj) -> decltype(obj->csr_dest, bool()) { return true; }
    static constexpr auto checkVariable_csr_weight(F *obj) -> decltype(obj->csr_weight, bool()) { return true; }
    static constexpr auto checkVariable_outDegree(F *obj) -> decltype(obj->outDegree, bool()) { return true; }
    static constexpr auto checkVariable_inDegree(F *obj) -> decltype(obj->inDegree, bool()) { return true; }
    static constexpr auto checkVariable_machine_offset(F *obj) -> decltype(obj->machine_offset, bool()) { return true; }
    static constexpr auto checkVariable_vertexValue_result(F *obj) -> decltype(obj->vertexValue_result, bool()) { return true; }

}; // end of class [AssertClass]

class BasicAssert
{
  public:
    static void checkPlatformAndThread()
    {
        static_assert(sizeof(size_t) == 8, "Our System Need 64-bits Platform");
        assert_msg(ThreadNum >= 2, "The Max Threads Must Large Than 2, but current is (%zu)", SCU64(ThreadNum));
        CPUInfo cpuInfo;
        cpuInfo.print();
        Msg_info("Used Thread: [%zu]", static_cast<uint64_t>(ThreadNum));
    }

    static void checkAlgorithm(Algorithm_type algorithm)
    {
        if (algorithm == Algorithm_type::PageRank)
        {
            if ((!std::is_same_v<vertex_data_type, float>)&&(!std::is_same_v<vertex_data_type, double>))
            {
                Msg_error("When run [%s], you need set the <edge_data_type> to float or double", Algorithm_type_name[SCI32(algorithm)]);
                exit(1);
            }
        }
        else
        {
            if ((std::is_same_v<vertex_data_type, float>) || (std::is_same_v<vertex_data_type, double>))
            {
                Msg_warn("When run [%s], you can set <edge_data_type> as uint32_t or uint64_t instead of float or double",
                         Algorithm_type_name[SCI32(algorithm)]);
                exit(1);
            }
        }
    }

    static void checkDataType(const size_t vertexNum, const size_t edgeNum)
    {

        if (vertexNum > std::numeric_limits<count_type>::max())
        {
            Msg_error("vertexNum need set the <count_type> to uint64_t");
            exit(1);
        }

        if (vertexNum > std::numeric_limits<vertex_id_type>::max())
        {
            Msg_error("vertexNum need set the <vertex_id_type> to uint64_t");
            exit(1);
        }

        if (edgeNum > std::numeric_limits<countl_type>::max())
        {
            Msg_error("edgeNum need set the <countl_type> to uint64_t");
            exit(1);
        }
    }

    static void checkEnemType(const OrderMethod_type &orderMethod, const Algorithm_type &algorithm, const ComputeEngine_type &computeEngine,
                              const GPU_memory_type &gpuMemory, const GPU_implement_type &gpuImple)
    {
        assert_msg(orderMethod < OrderMethod_type::MAX_VALUE, "You Input <orderMethod> Error");
        assert_msg(algorithm < Algorithm_type::MAX_VALUE, "You Input <algorithm> Error");
        assert_msg(computeEngine < ComputeEngine_type::MAX_VALUE, "You Input <computeEngine> Error");
        assert_msg(gpuMemory < GPU_memory_type::MAX_VALUE, "You Input <gpuMemory> Error");
        assert_msg(gpuImple < GPU_implement_type::MAX_VALUE, "You Input <gpuImple> Error");
    }
};
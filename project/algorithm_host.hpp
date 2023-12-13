#pragma once

#include "Basic/Thread/atomic_linux.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/Type/data_type.hpp"
#include "Basic/basic_include.cuh"
#include <limits>

/* *********************************************************************************************************
 * @description: BFS
 * *********************************************************************************************************/
namespace BFS_HOST_SPACE {

template <typename Graph> count_type BFS(Graph &graph, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end)
{
    count_type local_activeVertices = 0;
    // count_type noZeroOutDegreeNum = graph.noZeroOutDegreeNum;

    for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
    {
        vertex_id_type dest = graph.csr_dest[nbr_cur];

        if (graph.vertexValue_in[dest] == std::numeric_limits<vertex_data_type>::max())
        {
            graph.vertexValue_in[dest] = graph.level + 1;
            // if (dest < noZeroOutDegreeNum)
            graph.active.out().set_bit(dest);
            local_activeVertices += 1;
        }
    }

    return local_activeVertices;
}

} // namespace BFS_HOST_SPACE

/* *********************************************************************************************************
 * @description: SSSP
 * *********************************************************************************************************/
namespace SSSP_HOST_SPACE {

template <typename Graph> count_type SSSP(Graph &graph, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end)
{
    count_type local_activeVertices = 0;
    vertex_data_type srcVertexValue = graph.vertexValue_in[vertexId];
    // count_type noZeroOutDegreeNum = graph.noZeroOutDegreeNum;

    for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
    {
        // 获取dest的Offset
        vertex_id_type dest = graph.csr_dest[nbr_cur];
        edge_data_type weight = graph.csr_weight[nbr_cur];
        vertex_data_type distance = srcVertexValue + weight; //! 如果在这每次调用: graph.vertexValue_in[vertexId] 呢

        if (distance < graph.vertexValue_in[dest])
        {
            if (LinuxAtomic::write_min(&graph.vertexValue_in[dest], distance))
            {
                // if (dest < noZeroOutDegreeNum)
                graph.active.out().set_bit(dest);
                local_activeVertices += 1;
            }
        }
    }

    return local_activeVertices;
}

} // namespace SSSP_HOST_SPACE

/* *********************************************************************************************************
 * @description: WCC
 * *********************************************************************************************************/
namespace WCC_HOST_SPACE {

template <typename Graph> count_type WCC(Graph &graph, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end)
{
    count_type local_activeVertices = 0;
    vertex_data_type srcVertexValue = graph.vertexValue_in[vertexId];
    // count_type noZeroOutDegreeNum = graph.noZeroOutDegreeNum;

    for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
    {
        // 获取dest的Offset
        vertex_id_type dest = graph.csr_dest[nbr_cur];

        if (srcVertexValue < graph.vertexValue_in[dest])
        {
            if (LinuxAtomic::write_min(&graph.vertexValue_in[dest], srcVertexValue))
            {
                // if (dest < noZeroOutDegreeNum)
                graph.active.out().set_bit(dest);
                local_activeVertices += 1;
            }
        }
    }

    return local_activeVertices;
}

} // namespace WCC_HOST_SPACE

/* *********************************************************************************************************
 * @description: PR
 * *********************************************************************************************************/
namespace PR_HOST_SPACE {

template <typename Graph> count_type PageRank(Graph &graph, vertex_id_type vertexId, countl_type nbr_start, countl_type nbr_end)
{
    count_type local_activeVertices = 0;
    vertex_data_type srcVertexValue = graph.vertexValue_in[vertexId];

    for (countl_type nbr_cur = nbr_start; nbr_cur < nbr_end; nbr_cur++)
    {
        // 获取dest的Offset
        vertex_id_type dest = graph.csr_dest[nbr_cur];

        LinuxAtomic::write_add(&graph.vertexValue_out[dest], srcVertexValue);
    }

    return local_activeVertices; // 只用于占用返回位，返回值无实际意义
}

template <typename Graph> count_type Update(Graph &graph)
{
    count_type activeNum = 0;
#pragma omp parallel for reduction(+ : activeNum)
    for (vertex_id_type vertexId = 0; vertexId < graph.vertexNum; vertexId++)
    {
        graph.vertexValue_out[vertexId] = (1.0 - Alpha) / graph.vertexNum + Alpha * graph.vertexValue_out[vertexId];
        if (graph.outDegree[vertexId] > 0)
        {
            graph.vertexValue_out[vertexId] /= graph.outDegree[vertexId];
        }
        if ((graph.vertexValue_out[vertexId] - graph.vertexValue_in[vertexId]) > Tolerance)
        {
            graph.vertexValue_in[vertexId] = 0.0f;
            graph.active.out().set_bit(vertexId);
            activeNum += 1;
        }
        else
        {
            graph.vertexValue_in[vertexId] = 0.0f;
        }
    }
    return activeNum;
}

template <typename Graph> count_type UpdateCoprocess(Graph &graph)
{
    count_type activeNum = 0;
#pragma omp parallel for reduction(+ : activeNum)
    for (vertex_id_type vertexId = 0; vertexId < graph.noZeroOutDegreeNum; vertexId++)
    {
        graph.vertexValue_out[vertexId] = (1.0 - Alpha) / graph.vertexNum + Alpha * graph.vertexValue_out[vertexId];
        if (graph.outDegree[vertexId] > 0)
        {
            graph.vertexValue_out[vertexId] /= graph.outDegree[vertexId];
        }
        if ((graph.vertexValue_out[vertexId] - graph.vertexValue_in[vertexId]) > Tolerance)
        {
            graph.vertexValue_in[vertexId] = 0.0f;
            graph.active.out().set_bit(vertexId);
            activeNum += 1;
        }
        else
        {
            graph.vertexValue_in[vertexId] = 0.0f;
        }
    }
    return activeNum;
}

} // namespace PR_HOST_SPACE
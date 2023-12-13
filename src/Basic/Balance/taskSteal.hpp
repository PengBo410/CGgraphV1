#pragma once

#include "Basic/Balance/threadState.hpp"
#include "Basic/Bitmap/fixed_bitset.cuh"
#include "Basic/Console/console_V3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Type/data_type.hpp"
#include <cstddef>
#include <utility>

namespace Balance {

class TaskSteal
{

  private:
    ThreadState *threadState;
    Fixed_Bitset thread_working;

  public:
    ThreadState::ThreadState_type **thread_state;

    TaskSteal()
    {
        omp_set_dynamic(0);
        threadState = new ThreadState();
        thread_state = threadState->thread_state;

        thread_working.resize(ThreadNum);
    }

    /* ======================================================================================*
     *                              [taskSteal_twoStage]
     * ======================================================================================*/
    template <typename result_type = size_t, typename common_type = size_t>
    result_type taskSteal_twoStage(common_type work, std::function<void(size_t &, result_type &)> enCoderTask, size_t alignSize = 64)
    {
        taskSteal_splitTask<common_type>(work, alignSize);

        result_type totalWorkloads = 0;
#pragma omp parallel reduction(+ : totalWorkloads)
        {
            size_t thread_id = omp_get_thread_num();
            result_type totalTask_local = 0;

            /*************************************
             *   2.1.【VERTEX_WORKING】
             *************************************/
            while (true)
            {
                size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, 64);
                if (vertexId_current >= thread_state[thread_id]->end)
                    break;

                enCoderTask(vertexId_current,
                            totalTask_local); //[vertexId_current, vertexId_current + 64)

            } // end of [2.1.Vertex Working]

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, 64);
                    if (vertexId_current >= thread_state[threadId_help]->end)
                        break;

                    enCoderTask(vertexId_current,
                                totalTask_local); //[vertexId_current, vertexId_current + 64)
                }
            } // end of [2.2.VERTEX_STEALING]

            totalWorkloads += totalTask_local;
        }

        return totalWorkloads;
    }

    /* ======================================================================================*
     *                              [taskSteal_vertexLevel]
     * which is same as [taskSteal_twoStage], only rename [taskSteal_twoStage]
     * ======================================================================================*/
    size_t taskSteal_vertexLevel(size_t work, std::function<void(size_t &, size_t &)> enCoderTask, size_t alignSize = 64)
    {
        taskSteal_splitTask<size_t>(work, alignSize);

        size_t totalWorkloads = 0;
#pragma omp parallel reduction(+ : totalWorkloads)
        {
            size_t thread_id = omp_get_thread_num();
            size_t totalTask_local = 0;

            /*************************************
             *   2.1.【VERTEX_WORKING】
             *************************************/
            while (true)
            {
                size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, 64);
                if (vertexId_current >= thread_state[thread_id]->end)
                    break;

                enCoderTask(vertexId_current,
                            totalTask_local); //[vertexId_current, vertexId_current + 64)

            } // end of [2.1.Vertex Working]

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, 64);
                    if (vertexId_current >= thread_state[threadId_help]->end)
                        break;

                    enCoderTask(vertexId_current,
                                totalTask_local); //[vertexId_current, vertexId_current + 64)
                }
            } // end of [2.2.VERTEX_STEALING]

            totalWorkloads += totalTask_local;
        }

        return totalWorkloads;
    }

    /* ======================================================================================*
     *                              [taskSteal_twoStage]
     * ======================================================================================*/
    template <typename result_type = size_t, typename common_type = size_t>
    std::pair<result_type, result_type>
    taskSteal_twoStage_doubleReturn(common_type work, std::function<void(size_t &, result_type &, result_type &)> enCoderTask, size_t alignSize = 64)
    {
        taskSteal_splitTask<common_type>(work, alignSize);

        result_type totalWorkloads_first = 0;
        result_type totalWorkloads_second = 0;
#pragma omp parallel reduction(+ : totalWorkloads_first, totalWorkloads_second)
        {
            size_t thread_id = omp_get_thread_num();
            result_type totalTask_local_first = 0;
            result_type totalTask_local_second = 0;

            /*************************************
             *   2.1.【VERTEX_WORKING】
             *************************************/
            while (true)
            {
                size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, 64);
                if (vertexId_current >= thread_state[thread_id]->end)
                    break;

                enCoderTask(vertexId_current, totalTask_local_first, totalTask_local_second); //[vertexId_current, vertexId_current + 64)

            } // end of [2.1.Vertex Working]

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, 64);
                    if (vertexId_current >= thread_state[threadId_help]->end)
                        break;

                    enCoderTask(vertexId_current, totalTask_local_first, totalTask_local_second); //[vertexId_current, vertexId_current + 64)
                }
            } // end of [2.2.VERTEX_STEALING]

            totalWorkloads_first += totalTask_local_first;
            totalWorkloads_second += totalTask_local_second;
        }

        return std::make_pair(totalWorkloads_first, totalWorkloads_second);
    }

    /* ======================================================================================*
     *                              [taskSteal_splitTask]
     * ======================================================================================*/
    template <typename T> void taskSteal_splitTask(T &workSize, size_t alignSize = 1, bool fillWord = false)
    {
        size_t bitNum = 8 * sizeof(size_t);
        if (fillWord)
            alignSize = bitNum;
        T taskSize = workSize;
        for (count_type threadId = 0; threadId < ThreadNum; threadId++)
        {
            if (fillWord && WORD_MOD(taskSize) != 0)
                taskSize = (taskSize / bitNum + 1) * bitNum;
            thread_state[threadId]->start = (taskSize / ThreadNum) / alignSize * alignSize * threadId;
            thread_state[threadId]->cur = thread_state[threadId]->start;
            thread_state[threadId]->end = (taskSize / ThreadNum) / alignSize * alignSize * (threadId + 1);
            thread_state[threadId]->edgeDonate.vertex = 0;
            thread_state[threadId]->edgeDonate.edge_cur = 0;
            thread_state[threadId]->edgeDonate.edge_end = 0;
            thread_state[threadId]->edgeDonate.edge_socket = 0;
            if (threadId == (ThreadNum - 1))
                thread_state[threadId]->end = taskSize;
            thread_state[threadId]->status = ThreadState::VERTEX_WORKING;
        }
    }

    /* ======================================================================================*
     *                              [taskSteal_threeStage]
     * ======================================================================================*/
    template <typename Graph> count_type taskSteal_threeStage(Graph &graph, std::function<count_type(vertex_id_type, countl_type, countl_type)> push)
    {

        taskSteal_splitTask<count_type>(graph.vertexNum, static_cast<size_t>(VERTEXWORK_CHUNK));

        thread_working.fill();

        count_type activeVertices = 0;
#pragma omp parallel reduction(+ : activeVertices)
        {

            count_type thread_id = omp_get_thread_num();
            count_type local_activeVertices = 0;

            while (true)
            {
                size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
                if (vertexId_current >= thread_state[thread_id]->end)
                    break;
                size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                while (word != 0)
                {
                    if (word & 1)
                    {

                        countl_type nbr_start = graph.csr_offset[vertexId_current];
                        countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                        if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
                        {
                            thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                            thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                            thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

                            thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                            while (true)
                            {
                                size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                if (edge_current >= nbr_end)
                                    break;
                                countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                local_activeVertices += push(vertexId_current, edge_current, edge_end);
                            }

                            thread_state[thread_id]->status = ThreadState::VERTEX_WORKING;
                        }
                        else if ((nbr_end - nbr_start) > 0)
                        {
                            local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                        }
                    }
                    vertexId_current++;
                    word = word >> 1;
                } // end of while word
            }

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
                    if (vertexId_current >= thread_state[threadId_help]->end)
                        break;
                    size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = graph.csr_offset[vertexId_current];
                            countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                            // vertex_data_type source_level = graphNuma.vertexValue_numa[socketId_help][vertexId_current];//another
                            if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
                            {
                                thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                                thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                                thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
                                thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                                while (true)
                                {
                                    size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                    if (edge_current >= nbr_end)
                                        break;
                                    countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                    local_activeVertices += push(vertexId_current, edge_current, edge_end);
                                }

                                thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
                            }
                            else if ((nbr_end - nbr_start) > 0)
                            {
                                local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
            }

            /*************************************
             *   2.3.【EDGE WORKING】
             *************************************/
            size_t edge_working_count = 0;
            thread_working.clear_bit(thread_id);
            vertex_data_type vertexId_current;
            countl_type edge_end;
            size_t edge_current;
            while (!thread_working.empty())
            {
                edge_working_count++;
                for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset)
                {
                    count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                    while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
                    {
                        do
                        {
                            vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
                            edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
                            edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
                        }
                        while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current,
                                                             edge_current + EDGESTEAL_CHUNK));

                        if (edge_current >= edge_end)
                            break;
                        edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
                        local_activeVertices += push(vertexId_current, edge_current, edge_end);
                    }
                }
            }
            activeVertices += local_activeVertices;

        } // end of omp_parallel

        return activeVertices;

    } // end of func ttaskSteal_threeStage

    /* ======================================================================================*
     *                              [taskSteal_threeStage_cutEdge]
     * only for CPU/GPU co-process, which
     * change : countl_type nbr_end = graph.csr_offset[vertexId_current];
     * to     : countl_type nbr_start = graph.newNbrStart[vertexId_current];
     * ======================================================================================*/
    template <typename Graph>
    count_type taskSteal_threeStage_cutEdge(Graph &graph, std::function<count_type(vertex_id_type, countl_type, countl_type)> push)
    {
        taskSteal_splitTask<count_type>(graph.vertexNum, static_cast<size_t>(VERTEXWORK_CHUNK));
        thread_working.fill();

        count_type activeVertices = 0;
#pragma omp parallel reduction(+ : activeVertices)
        {

            count_type thread_id = omp_get_thread_num();
            count_type local_activeVertices = 0;

            while (true)
            {
                size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
                if (vertexId_current >= thread_state[thread_id]->end)
                    break;
                size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                while (word != 0)
                {
                    if (word & 1)
                    {
                        countl_type nbr_start = graph.newNbrStart[vertexId_current];
                        countl_type nbr_end = graph.csr_offset[vertexId_current + 1];

                        if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
                        {
                            thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                            thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                            thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

                            thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                            while (true)
                            {
                                size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                if (edge_current >= nbr_end)
                                    break;
                                countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                local_activeVertices += push(vertexId_current, edge_current, edge_end);
                            }

                            thread_state[thread_id]->status = ThreadState::VERTEX_WORKING;
                        }
                        else if ((nbr_end - nbr_start) > 0)
                        {
                            local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                        }
                    }
                    vertexId_current++;
                    word = word >> 1;
                } // end of while word
            }

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
                    if (vertexId_current >= thread_state[threadId_help]->end)
                        break;
                    size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = graph.newNbrStart[vertexId_current];
                            countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                            // vertex_data_type source_level = graphNuma.vertexValue_numa[socketId_help][vertexId_current];//another
                            if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
                            {
                                thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                                thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                                thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
                                thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                                while (true)
                                {
                                    size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                    if (edge_current >= nbr_end)
                                        break;
                                    countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                    local_activeVertices += push(vertexId_current, edge_current, edge_end);
                                }

                                thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
                            }
                            else if ((nbr_end - nbr_start) > 0)
                            {
                                local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
            }

            /*************************************
             *   2.3.【EDGE WORKING】
             *************************************/
            size_t edge_working_count = 0;
            thread_working.clear_bit(thread_id);
            vertex_data_type vertexId_current;
            countl_type edge_end;
            size_t edge_current;
            while (!thread_working.empty())
            {
                edge_working_count++;
                for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset)
                {
                    count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                    while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
                    {
                        do
                        {
                            vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
                            edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
                            edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
                        }
                        while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current,
                                                             edge_current + EDGESTEAL_CHUNK));

                        if (edge_current >= edge_end)
                            break;
                        edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
                        local_activeVertices += push(vertexId_current, edge_current, edge_end);
                    }
                }
            }
            activeVertices += local_activeVertices;

        } // end of omp_parallel

        return activeVertices;

    } // end of func ttaskSteal_threeStage

    /* ======================================================================================*
     *                              [taskSteal_threeStage_sink]
     * change : taskSteal_splitTask<count_type>(graph.vertexNum, static_cast<size_t>(VERTEXWORK_CHUNK));
     * to     :  taskSteal_splitTask<count_type>(graph.noZeroOutDegreeNum, static_cast<size_t>(VERTEXWORK_CHUNK));
     * ======================================================================================*/
    template <typename Graph>
    count_type taskSteal_threeStage_sink(Graph &graph, std::function<count_type(vertex_id_type, countl_type, countl_type)> push)
    {

        taskSteal_splitTask<count_type>(graph.noZeroOutDegreeNum, static_cast<size_t>(VERTEXWORK_CHUNK));

        thread_working.fill();

        count_type activeVertices = 0;
#pragma omp parallel reduction(+ : activeVertices)
        {

            count_type thread_id = omp_get_thread_num();
            count_type local_activeVertices = 0;

            while (true)
            {
                size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
                if (vertexId_current >= thread_state[thread_id]->end)
                    break;
                size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                while (word != 0)
                {
                    if (word & 1)
                    {

                        countl_type nbr_start = graph.csr_offset[vertexId_current];
                        countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                        if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
                        {
                            thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                            thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                            thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

                            thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                            while (true)
                            {
                                size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                if (edge_current >= nbr_end)
                                    break;
                                countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                local_activeVertices += push(vertexId_current, edge_current, edge_end);
                            }

                            thread_state[thread_id]->status = ThreadState::VERTEX_WORKING;
                        }
                        else if ((nbr_end - nbr_start) > 0)
                        {
                            local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                        }
                    }
                    vertexId_current++;
                    word = word >> 1;
                } // end of while word
            }

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
                    if (vertexId_current >= thread_state[threadId_help]->end)
                        break;
                    size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = graph.csr_offset[vertexId_current];
                            countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                            // vertex_data_type source_level = graphNuma.vertexValue_numa[socketId_help][vertexId_current];//another
                            if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
                            {
                                thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                                thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                                thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
                                thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                                while (true)
                                {
                                    size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                    if (edge_current >= nbr_end)
                                        break;
                                    countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                    local_activeVertices += push(vertexId_current, edge_current, edge_end);
                                }

                                thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
                            }
                            else if ((nbr_end - nbr_start) > 0)
                            {
                                local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
            }

            /*************************************
             *   2.3.【EDGE WORKING】
             *************************************/
            size_t edge_working_count = 0;
            thread_working.clear_bit(thread_id);
            vertex_data_type vertexId_current;
            countl_type edge_end;
            size_t edge_current;
            while (!thread_working.empty())
            {
                edge_working_count++;
                for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset)
                {
                    count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                    while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
                    {
                        do
                        {
                            vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
                            edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
                            edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
                        }
                        while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current,
                                                             edge_current + EDGESTEAL_CHUNK));

                        if (edge_current >= edge_end)
                            break;
                        edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
                        local_activeVertices += push(vertexId_current, edge_current, edge_end);
                    }
                }
            }
            activeVertices += local_activeVertices;

        } // end of omp_parallel

        return activeVertices;

    } // end of func taskSteal_threeStage

    /* ======================================================================================*
     *                              [taskSteal_threeStage_cutEdge]
     * only for CPU/GPU co-process, which
     * change : countl_type nbr_end = graph.csr_offset[vertexId_current];
     * to     : countl_type nbr_start = graph.newNbrStart[vertexId_current];
     *
     * change : taskSteal_splitTask<count_type>(graph.vertexNum, static_cast<size_t>(VERTEXWORK_CHUNK));
     * to     :  taskSteal_splitTask<count_type>(graph.noZeroOutDegreeNum, static_cast<size_t>(VERTEXWORK_CHUNK));
     * ======================================================================================*/
    template <typename Graph>
    count_type taskSteal_threeStage_cutEdge_sink(Graph &graph, std::function<count_type(vertex_id_type, countl_type, countl_type)> push)
    {
        taskSteal_splitTask<count_type>(graph.noZeroOutDegreeNum, static_cast<size_t>(VERTEXWORK_CHUNK));

        thread_working.fill();

        count_type activeVertices = 0;
#pragma omp parallel reduction(+ : activeVertices)
        {

            count_type thread_id = omp_get_thread_num();
            count_type local_activeVertices = 0;

            while (true)
            {
                size_t vertexId_current = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
                if (vertexId_current >= thread_state[thread_id]->end)
                    break;
                size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                while (word != 0)
                {
                    if (word & 1)
                    {
                        countl_type nbr_start = graph.csr_offset[vertexId_current];
                        countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                        if (vertexId_current < graph.processVertexNum_GPU)
                        {
                            nbr_start += graph.outDegree_ownDevice[vertexId_current];
                        }
                        assert_msg((nbr_start <= nbr_end), "error");

                        if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
                        {
                            thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                            thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                            thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

                            thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                            while (true)
                            {
                                size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                if (edge_current >= nbr_end)
                                    break;
                                countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                local_activeVertices += push(vertexId_current, edge_current, edge_end);
                            }

                            thread_state[thread_id]->status = ThreadState::VERTEX_WORKING;
                        }
                        else if ((nbr_end - nbr_start) > 0)
                        {
                            local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                        }
                    }
                    vertexId_current++;
                    word = word >> 1;
                } // end of while word
            }

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t vertexId_current = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
                    if (vertexId_current >= thread_state[threadId_help]->end)
                        break;
                    size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = graph.csr_offset[vertexId_current];
                            countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                            if (vertexId_current < graph.processVertexNum_GPU)
                            {
                                nbr_start += graph.outDegree_ownDevice[vertexId_current];
                            }
                            assert_msg((nbr_start <= nbr_end), "error");

                            // vertex_data_type source_level = graphNuma.vertexValue_numa[socketId_help][vertexId_current];//another
                            if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
                            {
                                thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                                thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                                thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
                                thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                                while (true)
                                {
                                    size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                    if (edge_current >= nbr_end)
                                        break;
                                    countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                    local_activeVertices += push(vertexId_current, edge_current, edge_end);
                                }

                                thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
                            }
                            else if ((nbr_end - nbr_start) > 0)
                            {
                                local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
            }

            /*************************************
             *   2.3.【EDGE WORKING】
             *************************************/
            size_t edge_working_count = 0;
            thread_working.clear_bit(thread_id);
            vertex_data_type vertexId_current;
            countl_type edge_end;
            size_t edge_current;
            while (!thread_working.empty())
            {
                edge_working_count++;
                for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset)
                {
                    count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                    while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
                    {
                        do
                        {
                            vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
                            edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
                            edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
                        }
                        while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current,
                                                             edge_current + EDGESTEAL_CHUNK));

                        if (edge_current >= edge_end)
                            break;
                        edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
                        local_activeVertices += push(vertexId_current, edge_current, edge_end);
                    }
                }
            }
            activeVertices += local_activeVertices;

        } // end of omp_parallel

        return activeVertices;

    } // end of func ttaskSteal_threeStage

    //>-------------------------------------------------------------------------------------------------------------------------------------------------

    /**********************************************************************************************************
     * @description: A more scability taskSteal under edge level
     * @param [Graph] &graph
     * @param [size_t] taskNum
     * @param [size_t] start
     * @param [std::function] push
     * @return [*]
     **********************************************************************************************************/
    template <typename Graph>
    count_type taskSteal_edgeLevel(Graph &graph, size_t taskNum, size_t start,
                                   std::function<count_type(vertex_id_type, countl_type, countl_type)> push)
    {
        taskSteal_splitTask<size_t>(taskNum, static_cast<size_t>(VERTEXWORK_CHUNK));
        thread_working.fill();

        count_type activeVertices = 0;
#pragma omp parallel reduction(+ : activeVertices)
        {

            count_type thread_id = omp_get_thread_num();
            count_type local_activeVertices = 0;

            while (true)
            {
                size_t atomic_index = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
                if (atomic_index >= thread_state[thread_id]->end)
                    break;
                size_t vertexId_current = atomic_index + start;
                size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                while (word != 0)
                {
                    if (word & 1)
                    {

                        countl_type nbr_start = graph.csr_offset[vertexId_current];
                        countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                        if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
                        {
                            thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                            thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                            thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

                            thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                            while (true)
                            {
                                size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                if (edge_current >= nbr_end)
                                    break;
                                countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                local_activeVertices += push(vertexId_current, edge_current, edge_end);
                            }

                            thread_state[thread_id]->status = ThreadState::VERTEX_WORKING;
                        }
                        else if ((nbr_end - nbr_start) > 0)
                        {
                            local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                        }
                    }
                    vertexId_current++;
                    word = word >> 1;
                } // end of while word
            }

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t atomic_index = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
                    if (atomic_index >= thread_state[threadId_help]->end)
                        break;
                    size_t vertexId_current = atomic_index + start;
                    size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = graph.csr_offset[vertexId_current];
                            countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                            // vertex_data_type source_level = graphNuma.vertexValue_numa[socketId_help][vertexId_current];//another
                            if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
                            {
                                thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                                thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                                thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
                                thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                                while (true)
                                {
                                    size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                    if (edge_current >= nbr_end)
                                        break;
                                    countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                    local_activeVertices += push(vertexId_current, edge_current, edge_end);
                                }

                                thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
                            }
                            else if ((nbr_end - nbr_start) > 0)
                            {
                                local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
            }

            /*************************************
             *   2.3.【EDGE WORKING】
             *************************************/
            size_t edge_working_count = 0;
            thread_working.clear_bit(thread_id);
            vertex_id_type vertexId_current;
            countl_type edge_end;
            size_t edge_current;
            while (!thread_working.empty())
            {
                edge_working_count++;
                for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset)
                {
                    count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                    while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
                    {
                        do
                        {
                            vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
                            edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
                            edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
                        }
                        while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current,
                                                             edge_current + EDGESTEAL_CHUNK));

                        if (edge_current >= edge_end)
                            break;
                        edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
                        local_activeVertices += push(vertexId_current, edge_current, edge_end);
                    }
                }
            }
            activeVertices += local_activeVertices;

        } // end of omp_parallel

        return activeVertices;

    } // end of func taskSteal_threeStage

    /**********************************************************************************************************
     * @description: A more scability taskSteal under edge level, The different with @function taskSteal_edgeLevel
     *               is change countl_type nbr_start = graph.csr_offset[vertexId_current];
     *                      to countl_type nbr_start = graph.outDegree_ownDevice[vertexId_current]
     * @param [Graph] &graph
     * @param [size_t] taskNum
     * @param [size_t] start
     * @param [std::function] push
     * @return [*]
     **********************************************************************************************************/
    template <typename Graph>
    count_type taskSteal_edgeLevel_edgeCut(Graph &graph, size_t taskNum, size_t start,
                                           std::function<count_type(vertex_id_type, countl_type, countl_type)> push)
    {
        taskSteal_splitTask<size_t>(taskNum, static_cast<size_t>(VERTEXWORK_CHUNK));
        thread_working.fill();

        count_type activeVertices = 0;
#pragma omp parallel reduction(+ : activeVertices)
        {

            count_type thread_id = omp_get_thread_num();
            count_type local_activeVertices = 0;

            while (true)
            {
                size_t atomic_index = __sync_fetch_and_add(&thread_state[thread_id]->cur, VERTEXSTEAL_CHUNK);
                if (atomic_index >= thread_state[thread_id]->end)
                    break;
                size_t vertexId_current = atomic_index + start;
                size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                while (word != 0)
                {
                    if (word & 1)
                    {
                        countl_type nbr_start = graph.csr_offset[vertexId_current];
                        countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                        if (vertexId_current < graph.processVertexNum_GPU)
                        {
                            nbr_start += graph.outDegree_ownDevice[vertexId_current];
                        }

                        if ((nbr_end - nbr_start) >= EDGESTEAL_THRESHOLD)
                        {
                            thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                            thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                            thread_state[thread_id]->edgeDonate.edge_end = nbr_end;

                            thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                            while (true)
                            {
                                size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                if (edge_current >= nbr_end)
                                    break;
                                countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                local_activeVertices += push(vertexId_current, edge_current, edge_end);
                            }

                            thread_state[thread_id]->status = ThreadState::VERTEX_WORKING;
                        }
                        else if ((nbr_end - nbr_start) > 0)
                        {
                            local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                        }
                    }
                    vertexId_current++;
                    word = word >> 1;
                } // end of while word
            }

            /*************************************
             *   2.2.【VERTEX_STEALING】
             *************************************/
            thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
            for (count_type steal_offset = 1; steal_offset < ThreadNum; steal_offset++)
            {
                count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                while (thread_state[threadId_help]->status != ThreadState::VERTEX_STEALING)
                {
                    size_t atomic_index = __sync_fetch_and_add(&thread_state[threadId_help]->cur, VERTEXSTEAL_CHUNK);
                    if (atomic_index >= thread_state[threadId_help]->end)
                        break;
                    size_t vertexId_current = atomic_index + start;
                    size_t word = graph.active.in().array[WORD_OFFSET(vertexId_current)];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            countl_type nbr_start = graph.csr_offset[vertexId_current];
                            countl_type nbr_end = graph.csr_offset[vertexId_current + 1];
                            if (vertexId_current < graph.processVertexNum_GPU)
                            {
                                nbr_start += graph.outDegree_ownDevice[vertexId_current];
                            }

                            if ((nbr_end - nbr_start) > EDGESTEAL_THRESHOLD)
                            {
                                thread_state[thread_id]->edgeDonate.vertex = vertexId_current;
                                thread_state[thread_id]->edgeDonate.edge_cur = nbr_start;
                                thread_state[thread_id]->edgeDonate.edge_end = nbr_end;
                                thread_state[thread_id]->status = ThreadState::EDGE_DONATE;

                                while (true)
                                {
                                    size_t edge_current = __sync_fetch_and_add(&thread_state[thread_id]->edgeDonate.edge_cur, EDGESTEAL_CHUNK);
                                    if (edge_current >= nbr_end)
                                        break;
                                    countl_type edge_end = (edge_current + EDGESTEAL_CHUNK >= nbr_end) ? nbr_end : (edge_current + EDGESTEAL_CHUNK);
                                    local_activeVertices += push(vertexId_current, edge_current, edge_end);
                                }

                                thread_state[thread_id]->status = ThreadState::VERTEX_STEALING;
                            }
                            else if ((nbr_end - nbr_start) > 0)
                            {
                                local_activeVertices += push(vertexId_current, nbr_start, nbr_end);
                            }
                        }
                        vertexId_current++;
                        word = word >> 1;
                    } // end of while word
                }
            }

            /*************************************
             *   2.3.【EDGE WORKING】
             *************************************/
            size_t edge_working_count = 0;
            thread_working.clear_bit(thread_id);
            vertex_id_type vertexId_current;
            countl_type edge_end;
            size_t edge_current;
            while (!thread_working.empty())
            {
                edge_working_count++;
                for (count_type steal_offset = 1; steal_offset < ThreadNum; ++steal_offset)
                {
                    count_type threadId_help = (thread_id + steal_offset) % ThreadNum;
                    while (thread_state[threadId_help]->status == ThreadState::EDGE_DONATE)
                    {
                        do
                        {
                            vertexId_current = thread_state[threadId_help]->edgeDonate.vertex;
                            edge_end = thread_state[threadId_help]->edgeDonate.edge_end;
                            edge_current = thread_state[threadId_help]->edgeDonate.edge_cur;
                        }
                        while (!__sync_bool_compare_and_swap(&thread_state[threadId_help]->edgeDonate.edge_cur, edge_current,
                                                             edge_current + EDGESTEAL_CHUNK));

                        if (edge_current >= edge_end)
                            break;
                        edge_end = (edge_current + EDGESTEAL_CHUNK >= edge_end) ? edge_end : (edge_current + EDGESTEAL_CHUNK);
                        local_activeVertices += push(vertexId_current, edge_current, edge_end);
                    }
                }
            }
            activeVertices += local_activeVertices;

        } // end of omp_parallel

        return activeVertices;

    } // end of func ttaskSteal_threeStage

}; // end of class [TaskSteal]

} // namespace Balance
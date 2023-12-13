#pragma once

#include "Basic/basic_include.cuh"
#include "Compute_degree.cuh"
#include "help.hpp"
#include "sortNbr.hpp"

/*******************************************************************************************************
 * 当我们在CSR_Result_type中添加了outDegree和inDegree的定义后， 决定将获取CSR的过程继续从
 * main中剥离出来，使得main.cuh更突出Enginer
 *
 * @param [csrResult]    求出的CSR
 * @param [graphName]    图的名称
 * @param [OrderMethod_type]  Reorder的方法
 * @param [has_csrFile]  是否具有CSR file， 如果没有则调用graphBinLoader, 需要更久的时间
 * @param [sortNbr]      是否对每个顶点的nbr排序(升序)，有些算法需要保证Nbr是有序的
 * @param [getOutDegree] 是否获取Graph的outDegree, 获取会增加当前函数的时间和CSR_Result_type的空间占用
 * @param [getInDegree]  是否获取Graph的inDegree, 获取会增加当前函数的时间和CSR_Result_type的空间占用
 * @param [graphRep]     有时候我们可能想求的是CSR或者CSC
 * @param [sink_check]   对于将sink点后置的Reorder算法,检查是否所有的sink vertices全部后置
 *
 * @return Reorder算法的old2new，如果是OrderMethod::NATIVE,则返回值为nullptr
 *
 * 2023-8-31
 *******************************************************************************************************/
vertex_id_type *getGraphCSR(CSR_Result_type &csrResult, std::string graphName, OrderMethod_type OrderMethod_type, bool has_csrFile,
                            bool sortNbr = true, bool getOutDegree = true, bool getInDegree = true,
                            GraphRepresentation graphRep = GraphRepresentation::CSR, bool sink_check = true)
{
    size_t vertices = 0;
    size_t edges = 0;
    vertex_id_type *old2new = nullptr;
    count_type zeroOutDegreeNum = 0;

    GraphFile_type graphFile = getGraphFile(graphName, OrderMethod_type);
    std::string file_old2new = graphFile.old2newFile;
    std::string file_addition = graphFile.addtitionFile;
    vertices = graphFile.vertices;
    edges = graphFile.edges;
    Msg_info("GraphName:[%s], |V| = %zu, |E| = %zu", graphName.c_str(), static_cast<uint64_t>(vertices), static_cast<uint64_t>(edges));
    BasicAssert::checkDataType(vertices, edges);

    if (has_csrFile)
    {
        std::string csrOffsetFile = graphFile.csrOffsetFile;
        std::string csrDestFile = graphFile.csrDestFile;
        std::string csrWeightFile = graphFile.csrWeightFile;

        csrResult.csr_offset = load_binFile<countl_type>(csrOffsetFile, static_cast<uint64_t>(vertices + 1));
        csrResult.csr_dest = load_binFile<vertex_id_type>(csrDestFile, static_cast<uint64_t>(edges));
        csrResult.csr_weight = load_binFile<edge_data_type>(csrWeightFile, static_cast<uint64_t>(edges));

        csrResult.vertexNum = vertices;
        csrResult.edgeNum = edges;
        Msg_finish("Using CSRFile Construct csrResult complete");

        if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            count_type *temp = load_binFile<count_type>(file_addition, 1);
            zeroOutDegreeNum = temp[0];
            csrResult.noZeroOutDegreeNum = csrResult.vertexNum - zeroOutDegreeNum;
            Msg_info("zeroOutDegreeNum = [%zu] (%.2lf%%)", static_cast<uint64_t>(zeroOutDegreeNum),
                     ((double)zeroOutDegreeNum / csrResult.vertexNum) * 100);

            if (sink_check)
            {
                assert_msg_clear((csrResult.csr_offset[csrResult.vertexNum - zeroOutDegreeNum] == csrResult.edgeNum),
                                 "(csrResult.csr_offset[csrResult.vertexNum - zeroOutDegreeNum] != csrResult.edgeNum) - (%zu != %zu)",
                                 static_cast<uint64_t>(csrResult.csr_offset[csrResult.vertexNum - zeroOutDegreeNum]),
                                 static_cast<uint64_t>(csrResult.edgeNum));

                omp_parallel_for(size_t i = (csrResult.vertexNum - zeroOutDegreeNum); i < csrResult.vertexNum; i++)
                {
                    countl_type degreeSize = csrResult.csr_offset[i + 1] - csrResult.csr_offset[i];
                    assert_msg(degreeSize == 0, "degreeSize != 0, vertexId = %zu, degreeSize = %zu", static_cast<uint64_t>(i),
                               static_cast<uint64_t>(degreeSize));
                }
                Msg_finish("Sink Vertices Check Finished !");
            }
        }
        else
        {
            csrResult.noZeroOutDegreeNum = csrResult.vertexNum;
        }
    }
    else
    {
        std::string filePath = graphFile.graphFile;
        SharedMemory::GraphBinReader graphBinReader(graphRep);
        Msg_info("GraphFile: %s", filePath.c_str());
        graphBinReader.group_load_directedGraph(filePath, vertices);
        graphBinReader.sort_nbr();
        // CSR
        graphBinReader.getStruct_csrResult(csrResult);
        Msg_info("Using GraphFile Construct csrResult complete");
        // graphBinReader.printAdjList();
        graphBinReader.clearAdjList();

        if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            count_type *temp = load_binFile<count_type>(file_addition, 1);
            zeroOutDegreeNum = temp[0];
            csrResult.noZeroOutDegreeNum = csrResult.vertexNum - zeroOutDegreeNum;
            Msg_info("zeroOutDegreeNum = [%u] (%.2f)", zeroOutDegreeNum, ((double)zeroOutDegreeNum / csrResult.vertexNum) * 100);

            if (sink_check)
            {
                omp_parallel_for(size_t i = (csrResult.vertexNum - zeroOutDegreeNum); i < csrResult.vertexNum; i++)
                {
                    countl_type degreeSize = csrResult.csr_offset[i + 1] - csrResult.csr_offset[i];
                    assert_msg(degreeSize == 0, "degreeSize != 0, vertexId = %zu, degreeSize = %zu", static_cast<uint64_t>(i),
                               static_cast<uint64_t>(degreeSize));
                }
                Msg_check("Sink Vertices Check Finished !");
            }
        }
    }

    if ((OrderMethod_type != OrderMethod_type::NATIVE))
        old2new = load_binFile<vertex_id_type>(file_old2new, vertices);

    if (sortNbr)
        nbrSort_taskSteal(csrResult);

    if (getOutDegree || getInDegree)
    {
        Compute_degree *computeDegree = new Compute_degree(csrResult);
        if (getOutDegree)
            csrResult.outDegree = computeDegree->getOutdegree();
        if (getInDegree)
            csrResult.inDegree = computeDegree->getIndegree();
    }

    return old2new;

clear:
    csrResult.clearCSR();
    if (old2new != nullptr)
        delete[] old2new;
    STOP;
}

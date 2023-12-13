#pragma once

#include "Basic/Console/console_V3.hpp"
#include "Basic/Graph/util.hpp"
#include "basic_struct.hpp"
#include <assert.h>
#include <string>

GraphFile_type getGraphFile(std::string graphName, OrderMethod_type OrderMethod_type = OrderMethod_type::NATIVE)
{
    GraphFile_type graphFile;

    /* ***************************************************************************
     *                               [cusha]
     * ***************************************************************************/
    if (graphName == "cusha")
    {
        graphFile.vertices = 8;
        graphFile.edges = 14;

        if (OrderMethod_type == OrderMethod_type::NATIVE)
        {
            graphFile.graphFile = "/home/omnisky/inputFile/cushaSsspInput_u32_u32.bin";
            graphFile.common_root = 0;

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/native_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/native_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/native_csrWeight_u32.bin";

            return graphFile;
        }
        else if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            graphFile.old2newFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_old2new.txt";
            graphFile.addtitionFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_addition.txt";

            graphFile.common_root = 0; // ?

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrWeight_u32.bin";

            return graphFile;
        }
        else
        {
            assert_msg(false, "Graph File [%s]  Not Support Current OrderMethod_type", graphName.c_str());
        }
    } // end of [cusha]

    /* ***************************************************************************
     *                               [enwiki]
     * ***************************************************************************/
    else if (graphName == "enwiki")
    {
        graphFile.vertices = 4206785;
        graphFile.edges = 101311614;

        if (OrderMethod_type == OrderMethod_type::NATIVE)
        {
            graphFile.graphFile = "/data/webgraph/bin/enwiki-2013-uint32t.bin";
            graphFile.common_root = 0;

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/native_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/native_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/native_csrWeight_u32.bin";

            return graphFile;
        }

        else if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            graphFile.old2newFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_old2new.txt";
            graphFile.addtitionFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_addition.txt";

            graphFile.common_root = 0; // ?

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrWeight_u32.bin";

            return graphFile;
        }
        else
        {
            assert_msg(false, "Graph File [%s]  Not Support Current OrderMethod_type", graphName.c_str());
        }

    } // end of [enwiki]

    /****************************************************************************
     *                               [gsh2015tpd]
     ****************************************************************************/
    else if (graphName == "gsh2015tpd")
    {
        graphFile.vertices = 30809122;
        graphFile.edges = 581245983;

        if (OrderMethod_type == OrderMethod_type::NATIVE)
        {
            graphFile.graphFile = "/data/webgraph/bin/gsh-2015-tpd-uint32t.bin";
            graphFile.common_root = 23841917;

            graphFile.csrOffsetFile = "/data/webgraph/bin/" + graphName + "/native_csrOffset_u32.bin";
            graphFile.csrDestFile = "/data/webgraph/bin/" + graphName + "/native_csrDest_u32.bin";
            graphFile.csrWeightFile = "/data/webgraph/bin/" + graphName + "/native_csrWeight_u32.bin";

            return graphFile;
        }
        else if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            graphFile.old2newFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_old2new.txt";
            graphFile.addtitionFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_addition.txt";

            graphFile.common_root = 0; // ?

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrWeight_u32.bin";

            return graphFile;
        }
        else
        {
            Msg_info("图文件[%s]不支持当前的OrderMethod", graphName.c_str());
            exit(1);
        }
    }

    /* ***************************************************************************
     *                               [twitter2010]
     * ***************************************************************************/
    else if (graphName == "twitter2010")
    {
        graphFile.vertices = 61578415;
        graphFile.edges = 1468364884;

        if (OrderMethod_type == OrderMethod_type::NATIVE)
        {
            graphFile.graphFile = "/data/webgraph/2021-6-2/twitter2010_hasEdge_uint32t.bin";
            graphFile.common_root = 892741;

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/native_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/native_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/native_csrWeight_u32.bin";

            return graphFile;
        }
        else if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            graphFile.old2newFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_old2new.txt";
            graphFile.addtitionFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_addition.txt";

            graphFile.common_root = 0; // ?

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrWeight_u32.bin";

            return graphFile;
        }
        else
        {
            assert_msg(false, "Graph File [%s]  Not Support Current OrderMethod_type", graphName.c_str());
        }

    } // end of [twitter2010]

    /* ***************************************************************************
     *                               [friendster]
     * ***************************************************************************/
    else if (graphName == "friendster")
    {
        graphFile.vertices = 124836180;
        graphFile.edges = 1806067135;

        if (OrderMethod_type == OrderMethod_type::NATIVE)
        {
            graphFile.graphFile = "/data/webgraph/bin/friendster/native_bin.txt";
            graphFile.common_root = 25689;

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/native_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/native_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/native_csrWeight_u32.bin";

            return graphFile;
        }
        else if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            graphFile.old2newFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_old2new.txt";
            graphFile.addtitionFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_addition.txt";

            graphFile.common_root = 0; // ?

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrWeight_u32.bin";

            return graphFile;
        }
        else
        {
            Msg_info("图文件[%s]不支持当前的OrderMethod", graphName.c_str());
            exit(1);
        }
    }

    /* ***************************************************************************
     *                               [weiBo]
     * ***************************************************************************/
    else if (graphName == "enwiki")
    {
        graphFile.vertices = 70067871;
        graphFile.edges = 2586065333;

        if (OrderMethod_type == OrderMethod_type::NATIVE)
        {
            graphFile.graphFile = "/data/webgraph/bin/enwiki-2013-uint32t.bin";
            graphFile.common_root = 0;

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/native_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/native_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/native_csrWeight_u32.bin";

            return graphFile;
        }

        else if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            graphFile.old2newFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_old2new.txt";
            graphFile.addtitionFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_addition.txt";

            graphFile.common_root = 0; // ?

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrWeight_u32.bin";

            return graphFile;
        }
        else
        {
            assert_msg(false, "Graph File [%s]  Not Support Current OrderMethod_type", graphName.c_str());
        }

    } // end of [enwiki]

    /****************************************************************************
     *                               [uk-2007-02]
     ****************************************************************************/
    else if (graphName == "uk-2007-02")
    {
        graphFile.vertices = 110123614;
        graphFile.edges = 3922799564;

        if (OrderMethod_type == OrderMethod_type::NATIVE)
        {
            graphFile.graphFile = "/data/webgraph/bin/uk-2007-02-uint32t.bin";

            graphFile.csrOffsetFile = "/data/webgraph/bin/" + graphName + "/native_csrOffset_u32.bin";
            graphFile.csrDestFile = "/data/webgraph/bin/" + graphName + "/native_csrDest_u32.bin";
            graphFile.csrWeightFile = "/data/webgraph/bin/" + graphName + "/native_csrWeight_u32.bin";

            graphFile.common_root = 0;

            return graphFile;
        }
        else if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            graphFile.old2newFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_old2new.txt";
            graphFile.addtitionFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_addition.txt";

            graphFile.common_root = 0; // ?

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrWeight_u32.bin";

            return graphFile;
        }
        else
        {
            Msg_info("The graph file [%s] not support OrderMethod", graphName.c_str());
            exit(1);
        }
    }

    /* ***************************************************************************
     *                               [uk-union]
     * ***************************************************************************/
    else if (graphName == "uk-union")
    {
        graphFile.vertices = 133633040;
        graphFile.edges = 5475109924;

        if (OrderMethod_type == OrderMethod_type::NATIVE)
        {
            graphFile.graphFile = "/home/pengjie/graph_data/uk-union/uk-union_u32_reGen.bin";
            graphFile.common_root = 0;

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/native_csrOffset_u64.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/native_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/native_csrWeight_u32.bin";

            return graphFile;
        }
        else if (OrderMethod_type == OrderMethod_type::CGgraphR)
        {
            graphFile.old2newFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_old2new.txt";
            graphFile.addtitionFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_addition.txt";

            graphFile.common_root = 0; // ?

            graphFile.csrOffsetFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrOffset_u32.bin";
            graphFile.csrDestFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrDest_u32.bin";
            graphFile.csrWeightFile = get_BaseGraphFile_path() + graphName + "/CGgraphR_csrWeight_u32.bin";

            return graphFile;
        }
        else
        {
            assert_msg(false, "Graph File [%s]  Not Support Current OrderMethod_type", graphName.c_str());
        }

    } // end of [uk-union]

    else
    {
        assert_msg(false, "Can not find the graph [%s]", graphName.c_str());
    }

    return graphFile;
}

CSR_Result_type getToyGraph(std::string graphName, OrderMethod_type OrderMethod_type = OrderMethod_type::NATIVE)
{
    CSR_Result_type csrResult;

    if (graphName == "toy_0")
    {
        countl_type *offset = new countl_type[13];
        offset[0] = 0;
        offset[1] = 3;
        offset[2] = 6;
        offset[3] = 6;
        offset[4] = 8;
        offset[5] = 8;
        offset[6] = 11;
        offset[7] = 15;
        offset[8] = 16;
        offset[9] = 21;
        offset[10] = 22;
        offset[11] = 24;
        offset[12] = 26;

        vertex_id_type *dest = new vertex_id_type[26];
        dest[0] = 1;
        dest[1] = 3;
        dest[2] = 7;
        dest[3] = 3;
        dest[4] = 5;
        dest[5] = 9;
        // 2
        dest[6] = 5;
        dest[7] = 6;
        // 4
        dest[8] = 7;
        dest[9] = 9;
        dest[10] = 11;
        dest[11] = 0;
        dest[12] = 5;
        dest[13] = 8;
        dest[14] = 9; // 6
        dest[15] = 5;
        dest[16] = 0;
        dest[17] = 2;
        dest[18] = 4;
        dest[19] = 9;
        dest[20] = 11; // 8
        dest[21] = 5;
        dest[22] = 2;
        dest[23] = 8;
        dest[24] = 2;
        dest[25] = 5;

        edge_data_type *weight = new edge_data_type[26];
        weight[0] = 1;
        weight[1] = 3;
        weight[2] = 7;
        weight[3] = 3;
        weight[4] = 5;
        weight[5] = 9;
        weight[6] = 5;
        weight[7] = 6;
        weight[8] = 7;
        weight[9] = 9;
        weight[10] = 11;
        weight[11] = 0;
        weight[12] = 5;
        weight[13] = 8;
        weight[14] = 9;
        weight[15] = 5;
        weight[16] = 0;
        weight[17] = 2;
        weight[18] = 4;
        weight[19] = 9;
        weight[20] = 11;
        weight[21] = 5;
        weight[22] = 2;
        weight[23] = 8;
        weight[24] = 2;
        weight[25] = 5;

        csrResult.vertexNum = 12;
        csrResult.edgeNum = 26;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }
    else if (graphName == "toy_1")
    {
        countl_type *offset = new countl_type[13];
        offset[0] = 0;
        offset[1] = 3;
        offset[2] = 6;
        offset[3] = 6;
        offset[4] = 8;
        offset[5] = 8;
        offset[6] = 11;
        offset[7] = 16;
        offset[8] = 17;
        offset[9] = 21;
        offset[10] = 24;
        offset[11] = 26;
        offset[12] = 28;

        vertex_id_type *dest = new vertex_id_type[28];
        dest[0] = 1;
        dest[1] = 3;
        dest[2] = 7;
        dest[3] = 3;
        dest[4] = 5;
        dest[5] = 9;
        dest[6] = 5;
        dest[7] = 6;
        dest[8] = 7;
        dest[9] = 9;
        dest[10] = 11;
        dest[11] = 0;
        dest[12] = 5;
        dest[13] = 8;
        dest[14] = 9;
        dest[15] = 10;
        dest[16] = 5;
        dest[17] = 0;
        dest[18] = 4;
        dest[19] = 9;
        dest[20] = 11;
        dest[21] = 5;
        dest[22] = 6;
        dest[23] = 8;
        dest[24] = 2;
        dest[25] = 8;
        dest[26] = 2;
        dest[27] = 5;

        edge_data_type *weight = new edge_data_type[28];
        weight[0] = 1;
        weight[1] = 3;
        weight[2] = 7;
        weight[3] = 3;
        weight[4] = 5;
        weight[5] = 9;
        weight[6] = 5;
        weight[7] = 6;
        weight[8] = 7;
        weight[9] = 9;
        weight[10] = 11;
        weight[11] = 0;
        weight[12] = 5;
        weight[13] = 8;
        weight[14] = 9;
        weight[15] = 5;
        weight[16] = 0;
        weight[17] = 2;
        weight[18] = 4;
        weight[19] = 9;
        weight[20] = 11;
        weight[21] = 5;
        weight[22] = 2;
        weight[23] = 8;
        weight[24] = 2;
        weight[25] = 5;
        weight[26] = 2;
        weight[27] = 5;

        csrResult.vertexNum = 12;
        csrResult.edgeNum = 28;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }
    else if (graphName == "toy_2")
    {
        countl_type *offset = new countl_type[13];
        offset[0] = 0;
        offset[1] = 5;
        offset[2] = 8;
        offset[3] = 12;
        offset[4] = 16;
        offset[5] = 20;
        offset[6] = 22;
        offset[7] = 24;
        offset[8] = 25;
        offset[9] = 28;
        offset[10] = 30;
        offset[11] = 30;
        offset[12] = 30;

        vertex_id_type *dest = new vertex_id_type[30];
        dest[0] = 1;
        dest[1] = 2;
        dest[2] = 3;
        dest[3] = 7;
        dest[4] = 8;
        dest[5] = 2;
        dest[6] = 4;
        dest[7] = 5;
        dest[8] = 0;
        dest[9] = 1;
        dest[10] = 3;
        dest[11] = 5;
        dest[12] = 0;
        dest[13] = 2;
        dest[14] = 4;
        dest[15] = 6;
        dest[16] = 1;
        dest[17] = 3;
        dest[18] = 9;
        dest[19] = 10;
        dest[20] = 2;
        dest[21] = 7; // 5
        dest[22] = 1;
        dest[23] = 8;  // 6
        dest[24] = 11; // 7
        dest[25] = 0;
        dest[26] = 5;
        dest[27] = 10; // 8
        dest[28] = 0;
        dest[29] = 11; // 9

        edge_data_type *weight = new edge_data_type[30];
        weight[0] = 1;
        weight[1] = 2;
        weight[2] = 3;
        weight[3] = 7;
        weight[4] = 8;
        weight[5] = 9;
        weight[6] = 5;
        weight[7] = 6;
        weight[8] = 7;
        weight[9] = 9;
        weight[10] = 11;
        weight[11] = 0;
        weight[12] = 5;
        weight[13] = 8;
        weight[14] = 9;
        weight[15] = 5;
        weight[16] = 0;
        weight[17] = 2;
        weight[18] = 4;
        weight[19] = 9;
        weight[20] = 11;
        weight[21] = 5;
        weight[22] = 2;
        weight[23] = 8;
        weight[24] = 2;
        weight[25] = 5;
        weight[26] = 2;
        weight[27] = 5;
        dest[28] = 0;
        dest[29] = 11;

        csrResult.vertexNum = 12;
        csrResult.edgeNum = 30;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }

    else if (graphName == "toy_3")
    {
        countl_type *offset = new countl_type[13];
        offset[0] = 0;
        offset[1] = 3;
        offset[2] = 6;
        offset[3] = 6;
        offset[4] = 8;
        offset[5] = 8;
        offset[6] = 9;
        offset[7] = 13;
        offset[8] = 14;
        offset[9] = 19;
        offset[10] = 20;
        offset[11] = 22;
        offset[12] = 24;

        vertex_id_type *dest = new vertex_id_type[24];
        dest[0] = 1;
        dest[1] = 3;
        dest[2] = 7; // 0
        dest[3] = 3;
        dest[4] = 5;
        dest[5] = 9; // 1
        // 2
        dest[6] = 5;
        dest[7] = 6; // 3
        // 4
        dest[8] = 0; // 5
        // dest[9] = 9; dest[10] = 11;

        dest[9] = 0;
        dest[10] = 5;
        dest[11] = 8;
        dest[12] = 9; // 6
        dest[13] = 5;
        dest[14] = 0;
        dest[15] = 2;
        dest[16] = 4;
        dest[17] = 9;
        dest[18] = 11; // 8
        dest[19] = 5;
        dest[20] = 2;
        dest[21] = 8; // 12
        dest[22] = 2;
        dest[23] = 5; // 11

        edge_data_type *weight = new edge_data_type[24];
        weight[0] = 1;
        weight[1] = 3;
        weight[2] = 7;
        weight[3] = 3;
        weight[4] = 5;
        weight[5] = 9;
        weight[6] = 5;
        weight[7] = 6;
        weight[8] = 7;
        weight[9] = 9;
        weight[10] = 11;
        weight[11] = 0;
        weight[12] = 5;
        weight[13] = 8;
        weight[14] = 9;
        weight[15] = 5;
        weight[16] = 0;
        weight[17] = 2;
        weight[18] = 4;
        weight[19] = 9;
        weight[20] = 11;
        weight[21] = 5;
        weight[22] = 2;
        weight[23] = 8;
        weight[24] = 2;
        weight[25] = 5;

        csrResult.vertexNum = 12;
        csrResult.edgeNum = 24;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }
    // 去掉3->5，新增0->5
    else if (graphName == "toy_4")
    {
        countl_type *offset = new countl_type[13];
        offset[0] = 0;
        offset[1] = 4;
        offset[2] = 7;
        offset[3] = 7;
        offset[4] = 8;
        offset[5] = 8;
        offset[6] = 9;
        offset[7] = 13;
        offset[8] = 14;
        offset[9] = 19;
        offset[10] = 20;
        offset[11] = 22;
        offset[12] = 24;

        vertex_id_type *dest = new vertex_id_type[24];
        dest[0] = 1;
        dest[1] = 3;
        dest[2] = 7;
        dest[3] = 5; // 0
        dest[4] = 3;
        dest[5] = 5;
        dest[6] = 9; // 1
        // 2
        dest[7] = 6; // 3
        // 4
        dest[8] = 0; // 5
        // dest[9] = 9; dest[10] = 11;

        dest[9] = 0;
        dest[10] = 5;
        dest[11] = 8;
        dest[12] = 9; // 6
        dest[13] = 5;
        dest[14] = 0;
        dest[15] = 2;
        dest[16] = 4;
        dest[17] = 9;
        dest[18] = 11; // 8
        dest[19] = 5;
        dest[20] = 2;
        dest[21] = 8; // 10
        dest[22] = 2;
        dest[23] = 5; // 11

        edge_data_type *weight = new edge_data_type[24];
        weight[0] = 1;
        weight[1] = 3;
        weight[2] = 7;
        weight[3] = 3;
        weight[4] = 5;
        weight[5] = 9;
        weight[6] = 5;
        weight[7] = 6;
        weight[8] = 7;
        weight[9] = 9;
        weight[10] = 11;
        weight[11] = 0;
        weight[12] = 5;
        weight[13] = 8;
        weight[14] = 9;
        weight[15] = 5;
        weight[16] = 0;
        weight[17] = 2;
        weight[18] = 4;
        weight[19] = 9;
        weight[20] = 11;
        weight[21] = 5;
        weight[22] = 2;
        weight[23] = 8;
        weight[24] = 2;
        weight[25] = 5;

        csrResult.vertexNum = 12;
        csrResult.edgeNum = 24;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }
    // 新增0->9,去掉8->0,去掉8->9
    else if (graphName == "toy_5")
    {
        countl_type *offset = new countl_type[13];
        offset[0] = 0;
        offset[1] = 5;
        offset[2] = 8;
        offset[3] = 8;
        offset[4] = 9;
        offset[5] = 9;
        offset[6] = 10;
        offset[7] = 14;
        offset[8] = 15;
        offset[9] = 18;
        offset[10] = 19;
        offset[11] = 21;
        offset[12] = 23;

        vertex_id_type *dest = new vertex_id_type[23];
        dest[0] = 1;
        dest[1] = 3;
        dest[2] = 7;
        dest[3] = 5;
        dest[4] = 8; // 0
        dest[5] = 3;
        dest[6] = 5;
        dest[7] = 9; // 1
        // 2
        dest[8] = 6; // 3
        // 4
        dest[9] = 0; // 5
        // dest[9] = 9; dest[10] = 11;

        dest[10] = 0;
        dest[11] = 5;
        dest[12] = 8;
        dest[13] = 9; // 6
        dest[14] = 5; // 7
        dest[15] = 2;
        dest[16] = 4;
        dest[17] = 11; // 8
        dest[18] = 5;
        dest[19] = 2;
        dest[20] = 8; // 10
        dest[21] = 2;
        dest[22] = 5; // 11

        edge_data_type *weight = new edge_data_type[23];
        weight[0] = 1;
        weight[1] = 3;
        weight[2] = 7;
        weight[3] = 3;
        weight[4] = 5;
        weight[5] = 9;
        weight[6] = 5;
        weight[7] = 6;
        weight[8] = 7;
        weight[9] = 9;
        weight[10] = 11;
        weight[11] = 0;
        weight[12] = 5;
        weight[13] = 8;
        weight[14] = 9;
        weight[15] = 5;
        weight[16] = 0;
        weight[17] = 2;
        weight[18] = 4;
        weight[19] = 9;
        weight[20] = 11;
        weight[21] = 5;
        weight[22] = 2;

        csrResult.vertexNum = 12;
        csrResult.edgeNum = 23;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }
    // 新增1->10,去掉1->3
    else if (graphName == "toy_6")
    {
        countl_type *offset = new countl_type[13];
        offset[0] = 0;
        offset[1] = 5;
        offset[2] = 8;
        offset[3] = 8;
        offset[4] = 9;
        offset[5] = 9;
        offset[6] = 10;
        offset[7] = 14;
        offset[8] = 15;
        offset[9] = 18;
        offset[10] = 19;
        offset[11] = 21;
        offset[12] = 23;

        vertex_id_type *dest = new vertex_id_type[23];
        dest[0] = 1;
        dest[1] = 3;
        dest[2] = 7;
        dest[3] = 5;
        dest[4] = 8; // 0
        dest[5] = 10;
        dest[6] = 5;
        dest[7] = 9; // 1
        // 2
        dest[8] = 6; // 3
        // 4
        dest[9] = 0; // 5
        // dest[9] = 9; dest[10] = 11;

        dest[10] = 0;
        dest[11] = 5;
        dest[12] = 8;
        dest[13] = 9; // 6
        dest[14] = 5; // 7
        dest[15] = 2;
        dest[16] = 4;
        dest[17] = 11; // 8
        dest[18] = 5;
        dest[19] = 2;
        dest[20] = 8; // 10
        dest[21] = 2;
        dest[22] = 5; // 11

        edge_data_type *weight = new edge_data_type[23];
        weight[0] = 1;
        weight[1] = 3;
        weight[2] = 7;
        weight[3] = 3;
        weight[4] = 5;
        weight[5] = 9;
        weight[6] = 5;
        weight[7] = 6;
        weight[8] = 7;
        weight[9] = 9;
        weight[10] = 11;
        weight[11] = 0;
        weight[12] = 5;
        weight[13] = 8;
        weight[14] = 9;
        weight[15] = 5;
        weight[16] = 0;
        weight[17] = 2;
        weight[18] = 4;
        weight[19] = 9;
        weight[20] = 11;
        weight[21] = 5;
        weight[22] = 2;

        csrResult.vertexNum = 12;
        csrResult.edgeNum = 23;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }

    // 新增6->4,去掉6->9
    else if (graphName == "toy_7")
    {
        countl_type *offset = new countl_type[13];
        offset[0] = 0;
        offset[1] = 5;
        offset[2] = 8;
        offset[3] = 8;
        offset[4] = 9;
        offset[5] = 9;
        offset[6] = 10;
        offset[7] = 14;
        offset[8] = 15;
        offset[9] = 18;
        offset[10] = 19;
        offset[11] = 21;
        offset[12] = 23;

        vertex_id_type *dest = new vertex_id_type[23];
        dest[0] = 1;
        dest[1] = 3;
        dest[2] = 7;
        dest[3] = 5;
        dest[4] = 8; // 0
        dest[5] = 10;
        dest[6] = 5;
        dest[7] = 9; // 1
        // 2
        dest[8] = 6; // 3   -----
        // 4
        dest[9] = 0; // 5
        // dest[9] = 9; dest[10] = 11;

        dest[10] = 0;
        dest[11] = 5;
        dest[12] = 8;
        dest[13] = 4; // 6
        dest[14] = 5; // 7
        dest[15] = 2;
        dest[16] = 4;
        dest[17] = 11; // 8
        dest[18] = 5;  // 9
        dest[19] = 2;
        dest[20] = 8; // 10
        dest[21] = 2;
        dest[22] = 5; // 11

        edge_data_type *weight = new edge_data_type[23];
        weight[0] = 1;
        weight[1] = 3;
        weight[2] = 7;
        weight[3] = 3;
        weight[4] = 5;
        weight[5] = 9;
        weight[6] = 5;
        weight[7] = 6;
        weight[8] = 7;
        weight[9] = 9;
        weight[10] = 11;
        weight[11] = 0;
        weight[12] = 5;
        weight[13] = 8;
        weight[14] = 9;
        weight[15] = 5;
        weight[16] = 0;
        weight[17] = 2;
        weight[18] = 4;
        weight[19] = 9;
        weight[20] = 11;
        weight[21] = 5;
        weight[22] = 2;

        csrResult.vertexNum = 12;
        csrResult.edgeNum = 23;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }
    else if (graphName == "toy_metis")
    {
        countl_type *offset = new countl_type[8];
        offset[0] = 0;
        offset[1] = 3;
        offset[2] = 6;
        offset[3] = 10;
        offset[4] = 14;
        offset[5] = 17;
        offset[6] = 20;
        offset[7] = 22;

        vertex_id_type *dest = new vertex_id_type[22];
        // 0
        dest[0] = 1;
        dest[1] = 2;
        dest[2] = 4;

        // 1
        dest[3] = 0;
        dest[4] = 2;
        dest[5] = 3;

        // 2
        dest[6] = 0;
        dest[7] = 1;
        dest[8] = 4;
        dest[9] = 3;

        // 3
        dest[10] = 1;
        dest[11] = 2;
        dest[12] = 5;
        dest[13] = 6;

        // 4
        dest[14] = 0;
        dest[15] = 2;
        dest[16] = 5;

        // 5
        dest[17] = 3;
        dest[18] = 4;
        dest[19] = 6;

        // 6
        dest[20] = 3;
        dest[21] = 5;

        edge_data_type *weight = new edge_data_type[22];
        // 0
        weight[0] = 1;
        weight[1] = 2;
        weight[2] = 1;

        // 1
        weight[3] = 1;
        weight[4] = 2;
        weight[5] = 1;

        // 2
        weight[6] = 2;
        weight[7] = 2;
        weight[8] = 3;
        weight[9] = 2;

        // 3
        weight[10] = 1;
        weight[11] = 2;
        weight[12] = 2;
        weight[13] = 5;

        // 4
        weight[14] = 1;
        weight[15] = 3;
        weight[16] = 2;

        // 5
        weight[17] = 2;
        weight[18] = 2;
        weight[19] = 6;

        // 6
        weight[20] = 5;
        weight[21] = 6;

        csrResult.vertexNum = 7;
        csrResult.edgeNum = 22;
        csrResult.csr_offset = offset;
        csrResult.csr_dest = dest;
        csrResult.csr_weight = weight;
    }
    else
    {
        assert_msg(false, "未找到对应的Toy Graph");
    }

    return csrResult;
}

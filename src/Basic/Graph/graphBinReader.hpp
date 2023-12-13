#pragma once

#include "Basic/Console/console_V3.hpp"
#include "Basic/Other/IO.hpp"
#include <tbb/concurrent_vector.h>

#define DATA_TYPE_CHECK        // Type Check
#define INDEGREE               // Whether to compute inDegree
#define GRAPH_BIN_READER_DEBUG // Debug

namespace SharedMemory {

class GraphBinReader
{
  private:
    count_type vertexNum;
    countl_type edgeNum;

    count_type edgeDataType_size; // sizeof of (edge)
    count_type nbr_size;          // sizeof of (vertex + edge)
    count_type edge_size;         //  sizeof of (srcVertex + edge + destVertex), used to get edgeNum

    // File Data
    uint64_t chunk_size = 1 << 20; //  lines from graphFile per read, stream size 64000,1<<20(1048576)
    uint64_t bytes_to_read;        // the total bytes of this graphFile
    uint64_t read_offset;
    struct EdgeUnit
    {
        vertex_id_type src;
        vertex_id_type dst;
        edge_data_type edge_data;
    } __attribute__((packed)); // Cancel align

    // Degree
    degree_type *outDegree;
#ifdef INDEGREE
    degree_type *inDegree;
#endif

    // Neighbor
    typedef std::pair<vertex_id_type, edge_data_type> nbr_pair_type; // first is dest，second is weight
    typedef tbb::concurrent_vector<nbr_pair_type> neighbor_type;     // https://github.com/oneapi-src/oneTBB#Concurrent_Operations
    tbb::concurrent_vector<neighbor_type> adjlist;                   // two-dim vector

    GraphRepresentation graphRepresentation;

    // csr
    bool csr = false; // Now, csr is build ?
    countl_type *csr_offset;
    vertex_id_type *csr_dest;
    edge_data_type *csr_weight;

    // csc
    bool csc = false; // Now, csr is build ?
    countl_type *csc_offset;
    vertex_id_type *csc_src;
    edge_data_type *csc_weight;

  public:
    GraphBinReader(GraphRepresentation graphRepresentation = GraphRepresentation::CSR)
        : vertexNum(0), edgeNum(0), edgeDataType_size(0), nbr_size(0), edge_size(0)
    {
        edgeDataType_size = sizeof(edge_data_type);            // sizeof of (edge)
        nbr_size = sizeof(vertex_id_type) + edgeDataType_size; // sizeof of (vertex + edge)
        edge_size = sizeof(vertex_id_type) + nbr_size;         // sizeof of (srcVertex + edge + destVertex)
        this->graphRepresentation = graphRepresentation;
    }

    /* *****************************************************************
     *  Func: Load Graph File
     *
     * @param [std::string path]  Graph File
     * @param [count_type]        vertexNum
     * *****************************************************************/
    void group_load_directedGraph(std::string path, count_type vertices)
    {
        vertexNum = vertices;
        uint64_t total_bytes = file_size(path.c_str());              // return the file size in bytes
        edgeNum = static_cast<countl_type>(total_bytes / edge_size); // get total edgeNum

        adjlist.resize(vertexNum);
        Msg_info("|total_bytes| = %zu, |edge_size| = %zu", total_bytes, static_cast<uint64_t>(edge_size));
        Msg_info("|v| = %zu, |E| = %zu", static_cast<uint64_t>(vertexNum), static_cast<uint64_t>(edgeNum));

        bytes_to_read = total_bytes; // Shared Memory need all read
        read_offset = 0;             // strat from the head of graph file

        // Whether CSR or CSC, outDegree is required, PageRank is required
        outDegree = new degree_type[vertexNum];
        memset(outDegree, 0, vertexNum * sizeof(degree_type));

        if (graphRepresentation == GraphRepresentation::CSR)
        {
#ifdef INDEGREE
            inDegree = new degree_type[vertexNum];
            memset(inDegree, 0, vertexNum * sizeof(degree_type));
#endif
        }
        else if (graphRepresentation == GraphRepresentation::CSC)
        {
            inDegree = new degree_type[vertexNum];
            memset(inDegree, 0, vertexNum * sizeof(degree_type));
        }

        // Start reading data - Open the file and locate the read location
        int fin = open(path.c_str(), O_RDONLY); // O_RDONLY: Open the file in read-only mode
        assert(lseek(fin, read_offset, SEEK_SET) == read_offset);

        // Start reading data
        double rate = 0.1;
        uint64_t read_bytes = 0; // Number of bytes read by the current machine
        EdgeUnit *read_edge_buffer = new EdgeUnit[chunk_size];
        while (read_bytes < bytes_to_read)
        {
            int64_t curr_read_bytes; // Number of bytes read this time
            // If the block size is larger than the storage capacity of array_temp, the block is read using the maximum storage capacity of array_temp
            if (bytes_to_read - read_bytes > edge_size * chunk_size)
            { // A chunk_size can store up to 2^20 edges
                curr_read_bytes = read(fin, read_edge_buffer, edge_size * chunk_size);
            }
            // If the size of the block is smaller than array_temp's storage, the actual amount is read
            else
            {
                curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
            }
            assert(curr_read_bytes >= 0);
            read_bytes += curr_read_bytes;

            // For Large Graph, We Can See The Progress
            if (edgeNum > 100000000)
            {
                if (static_cast<double>(curr_read_bytes) / static_cast<double>(bytes_to_read) > rate)
                {
                    rate = rate + 0.1;
                    Msg_info("Graph Bin Load Has Finsihed %.2lf %%", static_cast<double>(curr_read_bytes) / static_cast<double>(bytes_to_read) * 100);
                }
            }

            uint64_t curr_read_edges = curr_read_bytes / edge_size; // Number of edges that have been read

            //> Total Processing Logic
            omp_parallel_for(uint64_t e_i = 0; e_i < curr_read_edges; e_i++)
            {
                vertex_id_type src = read_edge_buffer[e_i].src;
                vertex_id_type dst = read_edge_buffer[e_i].dst;
                edge_data_type weight = read_edge_buffer[e_i].edge_data;
                assert_msg((src != dst), "Graph Bin Format Error");

                //> Get outDegree
#ifdef DATA_TYPE_CHECK
                if (std::numeric_limits<degree_type>::max() <= (outDegree[src]))
                {
                    assert_msg(false, "<degree_type> is not enough to hold [outDegree] array, outDegree[%zu] = %zu", static_cast<uint64_t>(src),
                               static_cast<uint64_t>(outDegree[src]));
                }
#endif
                __sync_fetch_and_add_8(&outDegree[src], static_cast<degree_type>(1)); //__sync_fetch_and_add_8(&outDegree[src], 1);//_8

                //> Get inDegree
#ifdef INDEGREE
#ifdef DATA_TYPE_CHECK
                if (std::numeric_limits<degree_type>::max() <= (inDegree[dst]))
                {
                    assert_msg(false, "<degree_type> is not enough to hold [inDegree] array");
                }
#endif
                __sync_fetch_and_add_8(&inDegree[dst], static_cast<degree_type>(1)); //__sync_fetch_and_add_8(&inDegree[dst], 1);//_8
#endif

                //> The Format Of [CSR] In Vector
                if (graphRepresentation == GraphRepresentation::CSR)
                {
                    adjlist[src].emplace_back(std::make_pair(dst, weight));
                }
                //> The Format Of [CSC] In Vector
                else if (graphRepresentation == GraphRepresentation::CSC)
                {
                    adjlist[dst].emplace_back(std::make_pair(src, weight));
                }
            }
        }
        delete[] read_edge_buffer;
        assert_msg(close(fin) == 0, "Graph Bin Close Error");
        Msg_finish("Graph Bin Load Finshed !");

#ifdef GRAPH_BIN_READER_DEBUG
        checkAdjList();
#endif
    } // end of func [group_load_directedGraph(...)]

    /* ********************************************************
     * Func: Check Adjest Is Result
     * ********************************************************/
    void checkAdjList()
    {
        omp_parallel_for(count_type i = 0; i < vertexNum; i++)
        {
            if (graphRepresentation == GraphRepresentation::CSR)
            {
                assert_msg(adjlist[i].size() == static_cast<size_t>(outDegree[i]), "In CSR, adjlist[i].size() != outDegree[i]");
            }
            else if (graphRepresentation == GraphRepresentation::CSC)
            {
                assert_msg(adjlist[i].size() == static_cast<size_t>(inDegree[i]), "In CSC, adjlist[i].size() != inDegree[i]");
            }
        }
        Msg_finish("Check Adjlist Finshed !");
    }

    inline static bool cmp(nbr_pair_type a, nbr_pair_type b) { return a.first < b.first; }

    /* ********************************************************
     * Func: Sort The Nbr Of Each Vertex
     *
     * @Note: We alos can use function [nbrSort_taskSteal]
     *        to achieve the high-performnance
     * ********************************************************/
    void sort_nbr()
    {
        omp_parallel_for(count_type i = 0; i < vertexNum; i++)
        {
            neighbor_type &nbr = adjlist[i];   // Must get the nbr's address
            sort(nbr.begin(), nbr.end(), cmp); // can also used lambda
        }
        Msg_finish("Sort_nbr Finished !");
#ifdef GRAPH_BIN_READER_DEBUG
        check_sortNbr();
#endif
    }

    /* ********************************************************
     * Func: Sort The Nbr Of Each Vertex
     *
     * @Note: We alos can use function [nbrSort_taskSteal]
     *        to achieve the high-performnance
     * ********************************************************/
    void check_sortNbr()
    {
        omp_parallel_for(count_type i = 0; i < vertexNum; i++)
        {
            neighbor_type nbr = adjlist[i];
            if (nbr.size() > 1)
            {
                for (degree_type j = 0; j < (nbr.size() - 1); j++)
                {
                    if (nbr[j].first >= nbr[j + 1].first)
                        printf("vertex = %d, nbr[j].first = %d,  nbr[j + 1].first=%d\n", i, nbr[j].first, nbr[j + 1].first);
                    assert(nbr[j].first < nbr[j + 1].first);
                }
            }
        }
        Msg_finish("Check sort_nbr Finished !");
    }

  private:
    /* ********************************************************
     * Func: Build The CSR
     * ********************************************************/
    void build_csr()
    {
        if (graphRepresentation != GraphRepresentation::CSR)
            assert_msg(false, "Current GraphRepresentation Can Not Build CSR");

        if (!csr)
        {
            if (adjlist.empty())
                assert_msg(false, "adjList Vector Empty");

            // Build csrOffset
            csr_offset = new countl_type[vertexNum + 1];
            csr_offset[0] = 0;
            for (count_type i = 1; i <= vertexNum; i++)
            {
                csr_offset[i] = csr_offset[i - 1] + static_cast<countl_type>(outDegree[i - 1]);
            }
            assert_msg(csr_offset[vertexNum] == edgeNum, "csr_offset[vertexNum] != edgeNum");

            // Build csrDest and csr_weight
            csr_dest = new vertex_id_type[edgeNum];
            csr_weight = new edge_data_type[edgeNum];
            omp_parallel_for(count_type i = 0; i < vertexNum; i++)
            {
                neighbor_type nbr = adjlist[i];
                countl_type offset = csr_offset[i];
                for (countl_type j = 0; j < nbr.size(); j++)
                {
                    csr_dest[offset + j] = nbr[j].first;
                    csr_weight[offset + j] = nbr[j].second;
                }
            }

            csr = true;

            // Here we change to the user manually free according to the demand
            // tbb::concurrent_vector<neighbor_type>().swap(adjlist);//Free tbb::concurrent_vector memory
        }
    }

    /* ********************************************************
     * Func: Build The CSC
     * ********************************************************/
    void build_csc()
    {
        if (graphRepresentation != GraphRepresentation::CSC)
            assert_msg(false, "Current GraphRepresentation Can Not Build CSC");

        if (!csc)
        {
            if (adjlist.empty())
                assert_msg(false, "adjList Vector Empty");

            // Build csc_Offset
            csc_offset = new countl_type[vertexNum + 1];
            csc_offset[0] = 0;
            for (count_type i = 1; i <= vertexNum; i++)
            {
                csc_offset[i] = csc_offset[i - 1] + static_cast<countl_type>(inDegree[i - 1]);
            }
            assert_msg(csc_offset[vertexNum] == edgeNum, "csc_offset[vertexNum] ！= edgeNum");

            // Build csc_Sr and csc_weight
            csc_src = new vertex_id_type[edgeNum];
            csc_weight = new edge_data_type[edgeNum];
            omp_parallel_for(count_type i = 0; i < vertexNum; i++)
            {
                neighbor_type nbr = adjlist[i];
                countl_type offset = csc_offset[i];
                for (countl_type j = 0; j < nbr.size(); j++)
                {
                    csc_src[offset + j] = nbr[j].first;
                    csc_weight[offset + j] = nbr[j].second;
                }
            }

            csc = true;

            // Here we change to the user manually free according to the demand
            // tbb::concurrent_vector<neighbor_type>().swap(adjlist);//Free tbb::concurrent_vector memory
        }
    }

  public:
    void getEdgeList_noWeight(EdgeList_noWeight_type &edgeList)
    {
        if (adjlist.size() == 0)
            assert_msg(false, "adjlist.size() = 0, adjList may be already cleared");
        if (edgeList.size() != 0)
            assert_msg(false, "edgeList.size() != 0, The initial edgeList must be the empty ");

        // The CSR offset may not be produced yet, so let's start with single threading
        timer t;
        for (count_type vertexId = 0; vertexId < vertexNum; vertexId++)
        {
            neighbor_type nbr = adjlist[vertexId];
            for (count_type nbrId = 0; nbrId < nbr.size(); nbrId++)
            {
                if (graphRepresentation == GraphRepresentation::CSR)
                {
                    edgeList.emplace_back(std::make_pair(vertexId, nbr[nbrId].first));
                }
                else if (graphRepresentation == GraphRepresentation::CSC)
                {
                    edgeList.emplace_back(std::make_pair(nbr[nbrId].first, vertexId));
                }
            }
        }
        assert_msg(edgeList.size() == edgeNum, "edgeList.size() != edgeNum");
        Msg_finish("getEdgeList(...) Used Time: %f (ms)", t.get_time_ms());
    }

    count_type getVertexNum()
    {
        if (vertexNum == 0)
        {
            printf("===> error:[vertexNum == 0] !\n");
            abort();
        }
        return vertexNum;
    }

    countl_type getEdgeNum()
    {
        if (edgeNum == 0)
        {
            printf("===> error:[edgeNum == 0] !\n");
            abort();
        }
        return edgeNum;
    }

    //===========================================================================
    //                                  [常用的函数]
    //===========================================================================
    void getStruct_csrResult(CSR_Result_type &csrResult)
    {
        build_csr();
        csrResult.vertexNum = this->vertexNum;
        csrResult.edgeNum = this->edgeNum;
        csrResult.csr_offset = this->csr_offset;
        csrResult.csr_dest = this->csr_dest;
        csrResult.csr_weight = this->csr_weight;
    }

    void getStruct_cscResult(CSC_Result_type &cscResult)
    {
        build_csc();
        cscResult.vertexNum = this->vertexNum;
        cscResult.edgeNum = this->edgeNum;
        cscResult.csc_offset = this->csc_offset;
        cscResult.csc_src = this->csc_src;
        cscResult.csc_weight = this->csc_weight;
    }

    void getResult_degree(Degree_type &degree_type)
    {
        degree_type.vertexNum = this->vertexNum;
        degree_type.edgeNum = this->edgeNum;
        degree_type.outDegree = this->outDegree;
#ifdef INDEGREE
        degree_type.inDegree = this->inDegree;
#endif
    }

    void clearAdjList()
    {
        tbb::concurrent_vector<neighbor_type>().swap(adjlist);
        Msg_finish("Adjlist Already Finished");
    }

#ifdef INDEGREE
    degree_type *getInDegree() { return inDegree; }
#endif

    void saveCSRtoFile(std::string csrOffsetFile, std::string csrDestFile, std::string csrWeightFile, bool overWrite = false)
    {
        // If the file does not exist or the specified permissions cannot be accessed, -1 is returned
        bool exist_csrOffsetFile = (access(csrOffsetFile.c_str(), F_OK) >= 0);
        bool exist_csrDestFile = (access(csrDestFile.c_str(), F_OK) >= 0);
        bool exist_csrWeightFile = (access(csrWeightFile.c_str(), F_OK) >= 0);

        if (exist_csrOffsetFile && exist_csrDestFile && exist_csrWeightFile && (!overWrite))
        {
            Msg_info("The CSR file already exists and will not be rewritten");
            // return;
        }

        build_csr();

        save_binArray<countl_type>(csrOffsetFile, csr_offset, vertexNum + 1);
        save_binArray<vertex_id_type>(csrDestFile, csr_dest, edgeNum);
        save_binArray<edge_data_type>(csrWeightFile, csr_weight, edgeNum);

        Msg_finish("SaveCSRtoFile Finsihed !");
    }

    //===========================================================================
    //                                  [Printf]
    //===========================================================================
    void printCSR()
    {
        std::stringstream ss;

        ss << "\ncsr_offset:\n";
        for (size_t i = 0; i < vertexNum + 1; i++)
        {
            ss << csr_offset[i] << " ";
        }

        ss << "\ncsr_dest:\n";
        for (size_t i = 0; i < edgeNum; i++)
        {
            ss << csr_dest[i] << " ";
        }

        ss << "\ncsr_weight:\n";
        for (size_t i = 0; i < edgeNum; i++)
        {
            ss << csr_weight[i] << " ";
        }

        // logstream(LOG_INFO) << ss.str() << std::endl;
    }

    void printCSC()
    {
        std::stringstream ss;

        ss << "\ncsc_offset:\n";
        for (size_t i = 0; i < vertexNum + 1; i++)
        {
            ss << csc_offset[i] << " ";
        }

        ss << "\ncsc_src:\n";
        for (size_t i = 0; i < edgeNum; i++)
        {
            ss << csc_src[i] << " ";
        }

        ss << "\ncsc_weight:\n";
        for (size_t i = 0; i < edgeNum; i++)
        {
            ss << csc_weight[i] << " ";
        }

        // logstream(LOG_INFO) << ss.str() << std::endl;
    }

    void printAdjList()
    {
        for (count_type i = 0; i < vertexNum; i++)
        {
            neighbor_type nbr = adjlist[i];
            printf("[%d]:", i);
            for (degree_type j = 0; j < nbr.size(); j++)
            {
                printf("%d(%d) ", nbr[j].first, nbr[j].second);
            }
            printf("\n");
        }
    }

    void print_sortGraph()
    {
        printf("print_sortGraph\n");
        for (count_type i = 0; i < vertexNum; i++)
        {
            neighbor_type nbr = adjlist[i];
            for (degree_type j = 0; j < nbr.size(); j++)
            {
                printf("%d %d %d\n", i, nbr[j].first, nbr[j].second);
            }
        }
    }

}; // end of Class [GraphBinReader]

} // namespace SharedMemory

// TODO
namespace Distribued {

class GraphBinReader
{
};

} // namespace Distribued

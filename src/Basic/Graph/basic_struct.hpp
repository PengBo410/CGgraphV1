#pragma once

#include "Basic/Type/data_type.hpp"
#include <cstddef>
#include <string>
#include <vector>

struct CSR_Result_type
{
    count_type vertexNum = 0;
    countl_type edgeNum = 0;
    countl_type *csr_offset = nullptr;
    vertex_id_type *csr_dest = nullptr;
    edge_data_type *csr_weight = nullptr;

    degree_type *outDegree = nullptr;
    degree_type *inDegree = nullptr;

    count_type noZeroOutDegreeNum = 0;

    void clearCSR()
    {
        if (csr_offset != nullptr)
        {
            delete[] csr_offset;
            csr_offset = nullptr;
        }

        if (csr_dest != nullptr)
        {
            delete[] csr_dest;
            csr_dest = nullptr;
        }

        if (csr_weight != nullptr)
        {
            delete[] csr_weight;
            csr_weight = nullptr;
        }

        if (outDegree != nullptr)
        {
            delete[] outDegree;
            outDegree = nullptr;
        }

        if (inDegree != nullptr)
        {
            delete[] inDegree;
            inDegree = nullptr;
        }
    }
};

struct CSC_Result_type
{
    count_type vertexNum = 0;
    countl_type edgeNum = 0;
    countl_type *csc_offset = nullptr;
    vertex_id_type *csc_src = nullptr;
    edge_data_type *csc_weight = nullptr;

    degree_type *outDegree = nullptr;
    degree_type *inDegree = nullptr;

    count_type noZeroOutDegreeNum = 0;

    void clearCSC()
    {
        if (csc_offset != nullptr)
        {
            delete[] csc_offset;
            csc_offset = nullptr;
        }

        if (csc_src != nullptr)
        {
            delete[] csc_src;
            csc_src = nullptr;
        }

        if (csc_weight != nullptr)
        {
            delete[] csc_weight;
            csc_weight = nullptr;
        }

        if (outDegree != nullptr)
        {
            delete[] outDegree;
            outDegree = nullptr;
        }

        if (inDegree != nullptr)
        {
            delete[] inDegree;
            inDegree = nullptr;
        }
    }
};

struct Coprocess_result_type
{
    CSR_Result_type csrResult_host;   // must GPU_MEM
    CSR_Result_type csrResult_device; // must ZERO_COPY
    vertex_id_type cutVertex = 0;
    bool isDeviceWhole = true;
    countl_type invokeGPU_edgeNum = 2500000;
    double invokeGPU_edgePct = 0.35;

    // degree_type *newNbrStart = nullptr;
    countl_type *csr_offset_CPUOwn = nullptr;
};

struct Degree_type
{
    count_type vertexNum;
    countl_type edgeNum;
    degree_type *outDegree;
    degree_type *inDegree;
};

typedef std::vector<std::pair<vertex_id_type, vertex_id_type>> BoundPointer_type;

enum SortDegree { OUTDEGREE, INDEGREES, DEGREE };

/****************************************************************************
 *                            [Graph Representation]
 ****************************************************************************/
enum GraphRepresentation { CSR, CSC };

/****************************************************************************
 *                              [GraphFile Struct]
 ****************************************************************************/
struct GraphFile_type
{
    std::string graphFile = "";
    size_t vertices = 0;

    size_t common_root = 0;
    size_t edges = 0;
    std::string old2newFile = "";
    std::string addtitionFile = "";

    std::string csrOffsetFile = "";
    std::string csrDestFile = "";
    std::string csrWeightFile = "";
};

typedef std::vector<std::pair<vertex_id_type, vertex_id_type>> EdgeList_noWeight_type;

//>-----------------------------------------------------------------------------------------------------------
//>-------------------------------------[We Update A New Version]---------------------------------------------
//>-----------------------------------------------------------------------------------------------------------

enum class Algorithm_type {
    BFS,
    SSSP,
    WCC,
    PageRank,

    MAX_VALUE // Use to get max value
};
const char *Algorithm_type_name[static_cast<uint32_t>(Algorithm_type::MAX_VALUE)] = {"BFS", "SSSP", "WCC", "PageRank"};
const char *Algorithm_type_help = "The Algorithm To Be Run: [0]:BFS, [1]:SSSP, [2]:WCC, [3]:PageRank";

enum class GPU_memory_type {
    GPU_MEM, // cudaMalloc
    UVM,     // cudaMemAdviseSetReadMostly

    MAX_VALUE // Use to get max value
};
const char *GPU_memory_type_name[static_cast<uint32_t>(GPU_memory_type::MAX_VALUE)] = {"GPU_MEM", "UVM"};
const char *GPU_memory_type_help = "The GPU Memory Type: [0]:GPU_MEM, [1]:UVM";

enum class GPU_implement_type {
    CAT,
    COALESCE_CHUNK,

    MAX_VALUE // Use to get max value
};
const char *GPU_implement_type_name[static_cast<uint32_t>(GPU_implement_type::MAX_VALUE)] = {"CAT", "COALESCE_CHUNK"};
const char *GPU_implement_type_help = "The GPU Implement Type: [0]:CAT, [1]:COALESCE_CHUNK";
enum class ComputeEngine_type {
    //^ Compute
    SINGLE_CORE, // Single-core
    MULTI_CORE,  // Multi-cores
    SINGLE_GPU,  // Single-GPU
    COOPERATION, // Co-CPU/GPU

    MAX_VALUE, // Use to get max value
};
const char *ComputeEngine_typee_name[static_cast<uint32_t>(ComputeEngine_type::MAX_VALUE)] = {"Single-core", "Multi-cores", "Single-GPU",
                                                                                              "Co-CPU/GPU"};
const char *ComputeEngine_type_help = "The Compute Engine: ([0]: Single-core, [1]Multi-cores, [2]:Single-GPU, [3]Co-CPU/GPU";

enum class DataEngine_type {
    //^ Data
    D2U,               // DirectedToUdirected, for (u->v) or (m<-n), save (u<->v) and (m<->n)
    D2U_SINGLE,        // DirectedToUdirected, for (u<->v) only save (u->v), stasify u < v
    REORDER,           // @see OrderMethod_type
    EXTRA_SUBGRAPH_VC, // Vertex-centric extra-subgraph
    EXTRA_SUBGRAPH_EC, // Edge-centric extra-subgraph
    SAVE_SINK,         // Save-sink
    MEASURE,           // Measure CPU/GPU

    MAX_VALUE // Use to get max value
};
const char *DataEngine_type_name[static_cast<uint32_t>(DataEngine_type::MAX_VALUE)] = {"DirectedToUdirected-one",
                                                                                       "DirectedToUdirected-two",
                                                                                       "Reorder",
                                                                                       "Vertex-centric extra-subgraph",
                                                                                       "Edge-centric extra-subgraph",
                                                                                       "Save-sink",
                                                                                       "Measure"};

enum class OrderMethod_type {
    NATIVE,
    RANDOM,
    CGgraphR,

    MAX_VALUE // Use to get max value
};
const char *OrderMethod_type_name[static_cast<uint32_t>(OrderMethod_type::MAX_VALUE)] = {"Native-order", "Random-order", "CGgraph-order"};
const char *OrderMethod_type_help = "The OrderMethod Can Be: [0]:Native, [1]:Random, [2]:CGgraph";

enum class GPU_state_type {
    INIT,
    WAIT,
    RUN,
    EXIT,

    MAX_VALUE // Use to get max value
};
const char *GPU_state_type_name[static_cast<uint32_t>(GPU_state_type::MAX_VALUE)] = {"INIT", "WAIT", "RUN", "EXIT"};

struct CheckInfo_type
{
    std::string graphName = "";
    Algorithm_type algorithm;
    size_t root = 0;
};
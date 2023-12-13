#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include <gflags/gflags.h>

//> gflags
DEFINE_string(graphName, "friendster", "The Graph Name");
DEFINE_int32(orderMethod, SCI32(OrderMethod_type::CGgraphR), OrderMethod_type_help);
DEFINE_int64(root, 25689, "The Scr For BFS/SSSP Or MaxIte For PageRank"); // TW: 892741
DEFINE_int32(algorithm, SCI32(Algorithm_type::PageRank), Algorithm_type_help);
DEFINE_int32(computeEngine, SCI32(ComputeEngine_type::COOPERATION), ComputeEngine_type_help);
DEFINE_int32(gpuMemory, SCI32(GPU_memory_type::GPU_MEM), GPU_memory_type_help);
DEFINE_int32(gpuImple, SCI32(GPU_implement_type::COALESCE_CHUNK), GPU_implement_type_help);
DEFINE_int32(runs, 1, "The Number Of Times The Algorithm Needs To Run");
// DEFINE_bool(has_csrFile, true, "Is There A Processed CSR Binary File Present (Improve The Graph File Loading)");
// DEFINE_bool(logResult, true, "Whether to check the processing results (Need Provided Check File)");
DEFINE_int32(useDeviceId, 0, "The GPU ID To Be Used");
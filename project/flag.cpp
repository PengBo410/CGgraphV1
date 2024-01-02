#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include <gflags/gflags.h>

//> gflags
DEFINE_string(graphName, "twitter2010", "The Graph Name");
DEFINE_int32(orderMethod, SCI32(OrderMethod_type::NATIVE), OrderMethod_type_help);
DEFINE_int64(root, 892741, "The Scr For BFS/SSSP Or MaxIte For PageRank");
DEFINE_int32(algorithm, SCI32(Algorithm_type::SSSP), Algorithm_type_help);
DEFINE_int32(computeEngine, SCI32(ComputeEngine_type::SINGLE_CORE), ComputeEngine_type_help);
DEFINE_int32(gpuMemory, SCI32(GPU_memory_type::GPU_MEM), GPU_memory_type_help);
DEFINE_int32(gpuImple, SCI32(GPU_implement_type::COALESCE_CHUNK), GPU_implement_type_help);
DEFINE_int32(runs, 1, "The Number Of Times The Algorithm Needs To Run");
DEFINE_int32(useDeviceId, 0, "The GPU ID To Be Used");
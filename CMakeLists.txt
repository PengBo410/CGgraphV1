cmake_minimum_required(VERSION 3.0.0)
project(CGgraph VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)



find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
	message("CUDA FOUND")
	set(CUDA_SOURCE_PROPERTY_FORMAT OBJ)
	set(CUDA_SEPARABLE_COMPILATION ON)
	include_directories(${CUDA_INCLUDE_DIRS})
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)	
	set(CUDA_NVCC_FLAGS --expt-extended-lambda -Xcompiler -fopenmp -Xcompiler -pthread -lgomp -Xcompiler -mavx2 -arch=sm_75;-O3;-G;-g;-std=c++17) #事实证明-O能提高代码2x多的执行时间
	link_directories($ENV{CUDA_PATH}/lib/x64)
else(${CUDA_FOUND})
	MESSAGE(STATUS "cuda not found!")
endif(${CUDA_FOUND})

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	message("[COMPLETE]:OPENMP FOUND")
	message("OpenMP_C_LIBRARIES = ${OpenMP_C_LIBRARIES}")
else(OPENMP_FOUND)
	message("[ERROR]:OPENMP NOT FOUND")
endif(OPENMP_FOUND)

set(CMAKELISTS_PARTNET_PATH ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKELISTS_PARTNET_PATH}/src/)
include_directories(${CMAKELISTS_PARTNET_PATH}/project/)
#include_directories(/home/omnisky/cpj_app/gflag/gflags-master/build/include/)

# Find TBB
find_library(TBB_LIBRARY tbb)
message("TBB_LIBRARIES: ${TBB_LIBRARY}")

find_package(gflags REQUIRED)

file(GLOB_RECURSE CU_FILE ${CMAKE_CURRENT_SOURCE_DIR}/project/*.cu)
file(GLOB_RECURSE CPP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/project/flag.cpp)
message("CU_FILE = ${CU_FILE}")
message("CPP_FILE = ${CPP_FILE}")
message("EFG_CPP_FILE = ${EFG_CPP_FILE}")
cuda_add_library(MY_CPP_CU_LIB SHARED ${CU_FILE} ${CPP_FILE})#${EFG_CPP_FILE}


CUDA_ADD_EXECUTABLE(CGgraph CGgraph.cu)
target_link_libraries(CGgraph hwloc numa cuda nvidia-ml pthread OpenMP::OpenMP_CXX tbb) #tbb  cuda nvidia-ml  -fopenmp
target_link_libraries(CGgraph /home/omnisky/cpj_app/gflag/gflags-master/build/lib/libgflags.a)
target_link_libraries(CGgraph MY_CPP_CU_LIB)

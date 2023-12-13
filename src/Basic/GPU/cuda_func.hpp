#pragma once

#include "Basic/basic_include.cuh"
#include <cassert>
#include <thrust/detail/type_traits.h>

// 参考资料：https://stackoverflow.com/questions/17399119/cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
__device__ __forceinline__ float fatomicMin(float *addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

template <typename T> __device__ __forceinline__ T cuda_atomicMin(T *addr, T value)
{
    if constexpr (thrust::detail::is_integral<T>::value)
    {
        return atomicMin(addr, value);
    }
    else if constexpr (thrust::detail::is_floating_point<T>::value)
    {
        return fatomicMin(addr, value);
    }
    else
    {
        printf("error");
        assert(false);
    }
}

inline __device__ uint32_t cuda_atomicAdd(uint32_t *address, uint32_t val) { return atomicAdd(address, val); }

inline __device__ uint64_t cuda_atomicAdd(uint64_t *address, uint64_t val)
{
    assert(sizeof(uint64_t) == sizeof(unsigned long long));
    return atomicAdd((unsigned long long *)address, (unsigned long long)val);
}

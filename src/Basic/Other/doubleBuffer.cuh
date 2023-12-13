#pragma once

#include "Basic/GPU/cuda_check.cuh"

template <typename T> struct DoubleBuffer
{

    T d_buffers[2];
    int selector;

    inline DoubleBuffer()
    {
        // d_buffers[0] = NULL;
        // d_buffers[1] = NULL;

        selector = 0;
    }

    inline DoubleBuffer(T &d_current, T &d_alternate)
    {
        // We used operate()to pass a reference
        d_buffers[0](d_current);
        d_buffers[1](d_alternate);

        selector = 0;
    }

    inline void setDoubleBuffer(T &d_current, T &d_alternate)
    {
        // We used operate()to pass a reference
        d_buffers[0](d_current);
        d_buffers[1](d_alternate);
    }

    inline T &Current() { return d_buffers[selector]; }
    inline T &Alternate() { return d_buffers[selector ^ 1]; } //^ : XOR, same is 0, different is 1

    inline void doubleBuffer_swap() { selector ^= 1; }

    inline T &in() { return d_buffers[selector]; }
    inline T &out() { return d_buffers[selector ^ 1]; } //^ : XOR, same is 0, different is 1
    inline void swap() { selector ^= 1; }
};

template <typename T> struct DoubleBuffer_array
{
    T *d_buffers[2];

    int selector;

    __host__ __device__ __forceinline__ DoubleBuffer_array()
    {
        selector = 0;
        d_buffers[0] = NULL;
        d_buffers[1] = NULL;
    }

    __host__ __device__ __forceinline__ DoubleBuffer_array(T *d_current,   ///< The currently valid buffer
                                                           T *d_alternate) ///< Alternate storage buffer of the same size as \p d_current
    {
        selector = 0;
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    __host__ __device__ __forceinline__ void setDoubleBuffer(T *d_current, T *d_alternate)
    {
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    __host__ __device__ __forceinline__ T *Current() { return d_buffers[selector]; }
    __host__ __device__ __forceinline__ T *Alternate() { return d_buffers[selector ^ 1]; }

    __host__ __device__ __forceinline__ T *in() { return d_buffers[selector]; }
    __host__ __device__ __forceinline__ T *out() { return d_buffers[selector ^ 1]; }
    __host__ __device__ __forceinline__ void swap() { selector ^= 1; }
};
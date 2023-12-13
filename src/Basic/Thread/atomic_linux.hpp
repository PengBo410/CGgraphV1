#pragma once

#include <assert.h>
#include <cstdint>

namespace LinuxAtomic {

template <class T> inline bool cas(T *ptr, T old_val, T new_val)
{
    if constexpr (sizeof(T) == 8)
    {
        return __sync_bool_compare_and_swap((long *)ptr, *((long *)&old_val), *((long *)&new_val));
    }
    else if constexpr (sizeof(T) == 4)
    {
        return __sync_bool_compare_and_swap((int *)ptr, *((int *)&old_val), *((int *)&new_val));
    }
    else
    {
        assert(false);
        return static_cast<bool>(0);
    }
}

template <class T> inline bool write_min(T *ptr, T val)
{
    volatile T curr_val;
    bool done = false;
    do
    {
        curr_val = *ptr;
    }
    while (curr_val > val && !(done = cas(ptr, curr_val, val)));
    return done;
}

template <class ET> inline bool write_max(ET *a, ET b)
{
    ET c;
    bool r = 0;
    do
    {
        c = *a;
    }
    while (c < b && !(r = cas(a, c, b)));
    return r;
}

template <class T> inline void write_add(T *ptr, T val)
{
    volatile T new_val, old_val;
    do
    {
        old_val = *ptr;
        new_val = old_val + val;
    }
    while (!cas(ptr, old_val, new_val));
}

template <class T> inline bool atomic_large(T *ptr, T val)
{
    volatile T curr_val;
    bool done = false;

    do
    {
        curr_val = *ptr;
        done = curr_val > val;
    }
    while (!cas(ptr, curr_val, curr_val));

    return done;
}

template <class T> inline bool atomic_largeEqu(T *ptr, T val)
{
    volatile T curr_val;
    bool done = false;

    do
    {
        curr_val = *ptr;
        done = curr_val >= val;
    }
    while (!cas(ptr, curr_val, curr_val));

    return done;
}

template <class T> inline bool atomic_smallEqu(T *ptr, T val)
{
    volatile T curr_val;
    bool done = false;

    do
    {
        curr_val = *ptr;
        done = curr_val <= val;
    }
    while (!cas(ptr, curr_val, curr_val));

    return done;
}

template <class T> inline int64_t atomic_length(T *ptr, T val)
{
    volatile T curr_val;
    int64_t length = 0;

    do
    {
        curr_val = *ptr;
        length = static_cast<int64_t>(curr_val) - static_cast<int64_t>(val);
    }
    while (!cas(ptr, curr_val, curr_val));

    return length;
}

} // namespace LinuxAtomic
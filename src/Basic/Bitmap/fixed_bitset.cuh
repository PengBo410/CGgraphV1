#pragma once

#include "Basic/Console/console_V3.hpp"
#include "Basic/GPU/cuda_check.cuh"
#include "Basic/Thread/omp_def.hpp"
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <numa.h>
#include <omp.h>
#include <stdint.h>

#include <cstdint>

typedef uint64_t bit_type;

#define SMART_LENGTH 1024000

#define BITSIZE 64
#define BIT_OFFSET(i) ((i) >> 6)
#define BIT_MOD(i) ((i)&0x3f) // 0x3f = 63

class Fixed_Bitset
{

  public:
    bit_type *array; // 数组
    bit_type len;    // 想要申请的len
    bit_type arrlen; // 数组的长度

    /* ***************************************************************************************************************************
     *                                            构造函数、析构函数与操作符                                                         * *
     * ***************************************************************************************************************************/

    // Fixed_Bitset(): array(NULL), len(0), arrlen(0) {}

    Fixed_Bitset() : array(NULL), len(0), arrlen(0) {}

    Fixed_Bitset(bit_type n) : array(NULL), len(0), arrlen(0) { setSize(n); }

    Fixed_Bitset(const Fixed_Bitset &db)
    {
        array = NULL;
        len = 0;
        arrlen = 0;
        *this = db;
    }

    ~Fixed_Bitset() { CUDA_CHECK(cudaFreeHost(array)); }

    /* ***************************************************************************************************************************
     *                                              Common Function
     * Func: fix_trailing_bits() 这会将块最后一位之后的剩余部分归零
     * ***************************************************************************************************************************/
    void fix_trailing_bits()
    {
        bit_type lastbits = BIT_MOD(len);
        if (lastbits == 0)
            return;
        array[arrlen - 1] &= ((bit_type(1) << lastbits) - 1);
    }

    void setSize(bit_type n)
    {
        if constexpr (sizeof(bit_type) != 8)
            assert_msg(false, "<bit_type> Only Support With 64 Bits");

        if (len != 0)
            assert_msg(false, "Fixed_Bitset Not Allow Set Size More Time");

        len = n;
        arrlen = BIT_OFFSET(n) + (BIT_MOD(n) > 0);
        CUDA_CHECK(cudaMallocHost((void **)&array, arrlen * sizeof(bit_type)));
        fix_trailing_bits();
        parallel_clear();
    }

    // 返回bitset的bit位数，即：len
    inline bit_type size() const { return len; }

    // 判断是否所有的bit位都为0
    inline bool empty() const
    {
        for (bit_type i = 0; i < arrlen; ++i)
            if (array[i])
                return false;
        return true;
    }

    inline bool parallel_empty() const
    {
        volatile bool flag = true;
#pragma omp parallel for shared(flag)
        for (bit_type i = 0; i < arrlen; ++i)
        {
            if (!flag)
                continue;
            if (array[i] == 0)
                flag = false;
        }
        return flag;
    }

    /* ***************************************************************************************************************************
     *                                                  fill and clear
     * Note: We provide single and parallel two versions
     * Performace:   parallel_clear() is better
     *               parallel_fill()  is better
     * ***************************************************************************************************************************/
    inline void clear()
    {
        for (bit_type i = 0; i < arrlen; ++i)
            array[i] = 0;
    }

    inline void clear_memset_() { memset((void *)array, 0, sizeof(bit_type) * arrlen); }

    inline void parallel_clear()
    {
        omp_parallel_for(bit_type i = 0; i < arrlen; ++i) { array[i] = 0; }
    }

    inline void fill()
    {
        for (bit_type i = 0; i < arrlen; ++i)
            array[i] = (bit_type)-1; // 将-1强转为size_t，也就是64个1，对应的十进制为：18446744073709551615
        fix_trailing_bits();
    }

    inline void parallel_fill()
    {
        omp_parallel_for(bit_type i = 0; i < arrlen; ++i) array[i] =
            (bit_type)-1; // 将-1强转为size_t，也就是64个1，对应的十进制为：18446744073709551615
        fix_trailing_bits();
    }

    /* ***************************************************************************************************************************
     *                                              Normal Function
     * ***************************************************************************************************************************/
    inline bool get(bit_type b) const
    {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        //(size_t(1) << size_t(bitpos))保证了第b位一定为1，所以原数据array[arrpos]中，若第b位为1则&的结果为1,返回true，否则为0，返回false；
        return array[arrpos] & (bit_type(1) << bit_type(bitpos));
    }

    inline bool set_bit(bit_type b)
    {
        // use CAS to set the bit
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type mask(bit_type(1) << bit_type(bitpos));
        // OR是或运算，A OR B的结果：当A、duB中只要有一个或者两个都为1时，结果果为1，否则为0；
        return __sync_fetch_and_or(array + arrpos, mask) & mask; // GCC里面的函数
    }

    inline bool set_bit_unsync(bit_type b)
    {
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type mask(bit_type(1) << bit_type(bitpos));
        bool ret = array[arrpos] & mask; // TODO：为什么相对于set_bit()是先&，在|=
        array[arrpos] |= mask;
        return ret;
    }

    inline bool clear_bit(bit_type b)
    {
        // use CAS to set the bit
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type test_mask(bit_type(1) << bit_type(bitpos));
        const bit_type clear_mask(~test_mask);
        return __sync_fetch_and_and(array + arrpos, clear_mask) & test_mask;
    }

    inline bool clear_bit_unsync(bit_type b)
    {
        // use CAS to set the bit
        bit_type arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        const bit_type test_mask(bit_type(1) << bit_type(bitpos));
        const bit_type clear_mask(~test_mask);
        bool ret = array[arrpos] & test_mask;
        array[arrpos] &= clear_mask;
        return ret;
    }

    // 将setbit的b位设置为新值[value]，并返回旧值。是线程安全的
    inline bool set(bit_type b, bool value)
    {
        if (value)
            return set_bit(b);
        else
            return clear_bit(b);
    }

    // 将setbit的b位设置为新值[value]，并返回旧值。不同于set()的是，此方法不使用原子操作，因此更快，但不够安全（可能存在多线程修改）
    inline bool set_unsync(bit_type b, bool value)
    {
        if (value)
            return set_bit_unsync(b);
        else
            return clear_bit_unsync(b);
    }

    // array中所有为1的总数
    bit_type popcount() const
    {
        bit_type ret = 0;
        for (bit_type i = 0; i < arrlen; ++i)
        {
            // 参考资料：https://blog.csdn.net/gaochao1900/article/details/5646211 (VC上也可以实现__builtin_popcountl)
            ret += __builtin_popcountl(array[i]); // 计算一个 64 位无符号整数有多少个位为1（_builtin_popcount()是计算32位的）
        }
        return ret;
    }

    bit_type parallel_popcount() const
    {
        bit_type ret = 0;
#pragma omp parallel for reduction(+ : ret)
        for (bit_type i = 0; i < arrlen; ++i)
        {
            // 参考资料：https://blog.csdn.net/gaochao1900/article/details/5646211 (VC上也可以实现__builtin_popcountl)
            ret += __builtin_popcountl(array[i]); // 计算一个 64 位无符号整数有多少个位为1（_builtin_popcount()是计算32位的）
        }
        return ret;
    }

    bit_type parallel_popcount(size_t end) const
    {
        assert(end <= len);
        size_t lasWord = (end + 63) / 64;
        bit_type ret = 0;
#pragma omp parallel for reduction(+ : ret)
        for (bit_type i = 0; i < (lasWord - 1); ++i)
        {
            // 参考资料：https://blog.csdn.net/gaochao1900/article/details/5646211 (VC上也可以实现__builtin_popcountl)
            ret += __builtin_popcountl(array[i]); // 计算一个 64 位无符号整数有多少个位为1（_builtin_popcount()是计算32位的）
        }
        size_t word = array[lasWord - 1];
        bit_type lastbits = BIT_MOD(end);
        if (lastbits != 0)
        {
            word &= ((bit_type(1) << lastbits) - 1);
            ret += __builtin_popcountl(word);
        }
        return ret;
    }

    // 返回包含第b位的word
    // word定义为特定下标的数组值，如：array[0],array[1]等
    inline size_t containing_word(size_t b)
    {
        size_t arrpos, bitpos;
        bit_to_pos(b, arrpos, bitpos);
        return array[arrpos];
    }

    /* ***************************************************************************************************************************
     *                                                  operator
     * Note: The Other operator Waiting...
     * ***************************************************************************************************************************/
    inline Fixed_Bitset &operator=(const Fixed_Bitset &db)
    {
        len = db.len;
        arrlen = db.arrlen;
        CUDA_CHECK(cudaMallocHost((void **)&array, arrlen * sizeof(bit_type)));
        memcpy(array, db.array, sizeof(bit_type) * arrlen);
        return *this;
    }

    // 操作符() - 这里我们采用operate()进行引用传递, Used In DoubleBuffer
    inline Fixed_Bitset &operator()(const Fixed_Bitset &db)
    {
        len = db.len;
        arrlen = db.arrlen;
        array = db.array;
        return *this;
    }

    /* ***************************************************************************************************************************
     *                                                 兼容dense_bitset
     * ***************************************************************************************************************************/
    void resize(bit_type n) { setSize(n); }

    void clear_memset() { parallel_clear(); }

    void clear_smart()
    {
        if (len <= SMART_LENGTH)
        {
            clear();
        }
        else
        {
            parallel_clear();
        }
    }

    void fill_smart()
    {
        if (len <= SMART_LENGTH)
        {
            fill();
        }
        else
        {
            parallel_fill();
        }
    }

  private:
    inline static void bit_to_pos(bit_type b, bit_type &arrpos, bit_type &bitpos)
    {
        // the compiler better optimize this...
        arrpos = BIT_OFFSET(b); // 当前bit位在数组中的位置，如：65位在数组的array[1]中
        bitpos = BIT_MOD(b);    // 相当于bitpos = b % (sizeof(size_t)),也就是在对应的数组中的位数
    }

}; // end of class [Fixed_Bitset]

#pragma once

#include <functional>
#include <iostream>

/* ********************************************************************************
 * Func: For Sort Container，return the target index
 *       return the first index satisfy >= target
 * Example: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ,14]
 *           2  3  4  5  6  7  8  9  10 11 11  11 11   12  13
 *         1.  target = 12, return 13
 *         2.  target = 11, return 9
 *         3.  target = 6,  return 4
 * ********************************************************************************/
template <typename Container_size_type, typename Element_type>
Container_size_type lower_bound(Container_size_type container_size, Element_type target,
                                std::function<Container_size_type(Container_size_type &)> container_element)
{
    Container_size_type binSearch_start = 0;
    Container_size_type binSearch_end = container_size;
    Container_size_type binSearch_mid = 0;
    while ((binSearch_end - binSearch_start) > 0)
    {
        Container_size_type _count2 = (binSearch_end - binSearch_start) >> 1;
        binSearch_mid = binSearch_start + _count2;

        if (container_element(binSearch_mid) >= target)
        {
            binSearch_end = binSearch_mid;
        }
        else
        {
            binSearch_start = binSearch_mid + 1;
        }
    }

    return binSearch_start;
}

template <typename T> std::pair<bool, int64_t> searchEleNum(T *arr, int64_t length, T target)
{
    int64_t start = 0;
    int64_t end = length - 1;
    while (start <= end)
    {
        int64_t mid = (end - start) / 2 + start;
        if (arr[mid] == target)
        {
            return std::make_pair(true, mid);
        }
        else if (arr[mid] < target)
        {
            start = mid + 1;
        }
        else if (arr[mid] > target)
        {
            end = mid - 1;
        }
    }

    if (start > 0)
        return std::make_pair(false, start);

    return std::make_pair(false, 0);
}

void test_searchEleNum()
{
    int nums[] = {1, 3, 5, 7, 9};

    for (int i = 0; i <= 10; i++)
    {
        std::pair<bool, int> index = searchEleNum(nums, 5, i);
        if (index.first)
        {
            std::cout << "[" << i << "] 前面共有 (" << (index.second + 1) << ")个数" << std::endl;
        }
        else
        {
            std::cout << "[" << i << "] 前面共有 (" << (index.second) << ")个数" << std::endl;
        }
    }
}

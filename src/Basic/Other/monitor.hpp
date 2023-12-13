#pragma once

#include "Basic/Console/console_V3.hpp"
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <sys/resource.h>
#include <unistd.h>

class Monitor
{
  private:
    struct rusage start_usage;
    struct rusage end_usage;

    size_t start_vMemory = 0;
    size_t end_vMemory = 0;

  public:
    Monitor()
    {
        start_getUsage();
        start_vMemory = getCurrentVirtualMemorySize();
    }

    // 以KB为单位
    size_t getCur_maxPhysicalMemory_KB()
    {
        end_endUsage();
        return static_cast<size_t>(end_usage.ru_maxrss);
    }

    // free 会减少虚拟内存
    size_t getCur_virualMemory_KB()
    {
        end_getVmemory();
        return (end_vMemory - start_vMemory) / 1024;
    }

    void start_getUsage() { getrusage(RUSAGE_SELF, &start_usage); }
    void end_endUsage() { getrusage(RUSAGE_SELF, &end_usage); }
    void start_getVmemory() { start_vMemory = getCurrentVirtualMemorySize(); }
    void end_getVmemory() { end_vMemory = getCurrentVirtualMemorySize(); }

  private:
    std::size_t getCurrentVirtualMemorySize()
    {
        std::ifstream stat_stream("/proc/self/statm");
        std::size_t size;
        stat_stream >> size;
        return size * sysconf(_SC_PAGESIZE); // sysconf(_SC_PAGESIZE) 获取每一页的大小
    }

}; // namespace Monitor
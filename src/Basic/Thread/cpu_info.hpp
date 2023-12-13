#pragma once

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <numa.h>
#include <vector>

/**
 *  用法：
 *     CPUInfo cpuinfo;
 * 	   cpuinfo.print();
 */

class CPUInfo
{

  private:
    int socketNum = 0;
    int coreNum = 0;
    int threadNum = 0;

    std::vector<std::pair<int, int>> coreId_threadId;
    std::vector<int> socketId_vec;
    std::vector<int> threadId_vec;
    std::vector<int> coreId_vec;

    int max_socketId = -1;
    int max_coreId = -1;
    int max_threadId = -1;

    struct Util
    {
        int socketId_;
        int threadId_;
        int coreId_;
    };
    std::vector<Util> util_vec;

    // std::vector < std::vector < std::vector <int> > > socketId_coreId_threadId;
  public:
    CPUInfo() { readCpuInfo(); }

  private:
    void readCpuInfo(std::string inputFile = "/proc/cpuinfo")
    {
        std::vector<std::vector<int>> coreId_threadId; // coreId与threadId的对应关系

        std::vector<std::vector<std::string>> file2vector;

        std::ifstream in_file(inputFile.c_str(), std::ios_base::in); // 以二进制读的方式打开,并写入内存
        if (!in_file.good())
        {
            std::cout << "[readCpuInfo]Error opening in-file: " << inputFile << std::endl;
            assert(false);
        }

        // 将文件输出到vector
        std::string lineString;
        std::vector<std::string> temp;
        while (in_file.good() && !in_file.eof())
        {

            std::getline(in_file, lineString);

            // 用空行来划分
            if (!lineString.empty()) // 不为空行
            {
                temp.push_back(lineString);
            }
            else
            {
                if (temp.size() != 0) // 避免文件的最后一个空行
                {
                    file2vector.push_back(temp);
                    temp.clear();
                }
            }
        }

        in_file.close();

        // 依次解析
        for (size_t blockId = 0; blockId < file2vector.size(); blockId++)
        {
            std::vector<std::string> block = file2vector[blockId];
            for (size_t lineId = 0; lineId < block.size(); lineId++)
            {
                std::string line = block[lineId];

                int socketId = parseSocketId(line);
                if (socketId != -1)
                    socketId_vec.push_back(socketId);
                if (socketId > max_socketId)
                    max_socketId = socketId;

                int coreId = parsePhysicalCore(line);
                if (coreId != -1)
                    coreId_vec.push_back(coreId);
                if (coreId > max_coreId)
                    max_coreId = coreId;

                int threadId = parseLogicalCore(line);
                if (threadId != -1)
                    threadId_vec.push_back(threadId);
                if (threadId > max_threadId)
                    max_threadId = threadId;
            }
        }

        // 构建关系
        assert(socketId_vec.size() == coreId_vec.size());
        assert(socketId_vec.size() == threadId_vec.size());

        for (size_t i = 0; i < socketId_vec.size(); i++)
        {
            Util util;
            util.socketId_ = socketId_vec[i];
            util.coreId_ = coreId_vec[i];
            util.threadId_ = threadId_vec[i];

            util_vec.push_back(util);
        }

        // 排序
        sort(util_vec.begin(), util_vec.end(),
             [](const Util &a, const Util &b)
             {
            if (a.socketId_ < b.socketId_)
            {
                return true;
            }
            else if (a.socketId_ == b.socketId_)
            {
                if (a.coreId_ < b.coreId_)
                {
                    return true;
                }
                else if (a.coreId_ == b.coreId_)
                {
                    if (a.threadId_ <= b.threadId_)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        });

        // 准备输出变量
        std::vector<int> socketId_vec_temp;
        std::vector<int> coreId_vec_temp;
        std::vector<int> threadId_vec_temp;

        for (size_t i = 0; i < util_vec.size(); i++)
        {
            Util util = util_vec[i];
            socketId_vec_temp.push_back(util.socketId_);
            coreId_vec_temp.push_back(util.coreId_);
            threadId_vec_temp.push_back(util.threadId_);
        }

        socketId_vec_temp.erase(unique(socketId_vec_temp.begin(), socketId_vec_temp.end()), socketId_vec_temp.end());
        coreId_vec_temp.erase(unique(coreId_vec_temp.begin(), coreId_vec_temp.end()), coreId_vec_temp.end());
        threadId_vec_temp.erase(unique(threadId_vec_temp.begin(), threadId_vec_temp.end()), threadId_vec_temp.end());

        socketNum = socketId_vec_temp.size();
        coreNum = coreId_vec_temp.size();
        threadNum = threadId_vec_temp.size();
    }

    int parseLogicalCore(std::string line)
    {
        int pro = -1;
        // 寻找logical core，也就是threadNum
        if (line.find("processor", 0) != std::string::npos)
        {
            // 寻找 ": "
            int index = line.find_last_of(": ");
            if (index != -1)
            {
                std::string value = line.substr(index); // 截取s中从从pos开始（包括0）到末尾的所有字符的子串，并返回
                pro = atoi(value.c_str());
            }
        }
        return pro;
    }

    int parsePhysicalCore(std::string line)
    {
        int core = -1;
        // 寻找physical core，也就是coreNum
        if (line.find("core id", 0) != std::string::npos)
        {
            // 寻找 ": "
            int index = line.find_last_of(": ");
            if (index != -1)
            {
                std::string value = line.substr(index); // 截取s中从从pos开始（包括0）到末尾的所有字符的子串，并返回
                core = atoi(value.c_str());
            }
        }
        return core;
    }

    int parseSocketId(std::string line)
    {
        int socketId = -1;
        // 寻找physical core，也就是coreNum
        if (line.find("physical id", 0) != std::string::npos)
        {
            // 寻找 ": "
            int index = line.find_last_of(": ");
            if (index != -1)
            {
                std::string value = line.substr(index); // 截取s中从从pos开始（包括0）到末尾的所有字符的子串，并返回
                socketId = atoi(value.c_str());
            }
        }
        return socketId;
    }

  public:
    int getSocketNum() { return socketNum; }

    int getCoreNum() { return coreNum; }

    int getThreadNum() { return threadNum; }

    bool isOpenHybirdThread() { return (coreNum == threadNum) ? false : true; }

    void print()
    {
        printf("Socket num = {%d}, Core num = [%d], Thread num = (%d), isOpenHybirdThread = *%d*\n", socketNum, coreNum, threadNum,
               isOpenHybirdThread());
        if (isOpenHybirdThread())
        {
            printf("***************************************************************\n");
            printf("*                        ---------------                      *\n");
            printf("*                        |             |                      *\n");
            printf("*                        |Hybird-Thread|                      *\n");
            printf("*                        |             |                      *\n");
            printf("*                        ---------------                      *\n");
            printf("***************************************************************\n");
        }
    }

    void print_socket_core_thread()
    {
        for (size_t i = 0; i < util_vec.size(); i++)
        {
            Util util = util_vec[i];
            printf("%zu: socket{%d} - core[%d] - threadId(%d)\n", i, util.socketId_, util.coreId_, util.threadId_);
        }
    }
};

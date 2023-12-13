#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <assert.h>
#include <fstream>
#include <sstream> // std::stringstream
#include <string>

#include "Basic/Console/console_V3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Thread/omp_def.hpp"
#include "Basic/Time/time.hpp"
#include "Basic/Type/data_type.hpp"
#include "tqdm.hpp"

#define USE_BAR_TQDM 1  // 是否使用进度条
#define BAR_THRESHOLD 1 // 10000000

/* ***************************************************************************************
 * Func: Get the Total File Size In Bytes
 *
 * @param [std::string filename]  The Graph File
 * ***************************************************************************************/
inline uint64_t file_size(std::string filename)
{
    struct stat st;
    assert_msg(stat(filename.c_str(), &st) == 0, "Bin File [%s] Not Exist", filename.c_str());
    return st.st_size;
}

/* ***************************************************************************************
 * Func: Load The Graph File To Array
 *
 * @param [std::string path]       The Array File
 * @param [uint64_t length]        The number of elements
 *
 * @return The Array Of Graph File
 * ***************************************************************************************/
template <typename T> T *load_binFile(std::string path, uint64_t length)
{
    uint64_t chunk_size = 1 << 20; // lines from graphFile per read, stream size 64000,1<<20(1048576)
    uint64_t total_bytes = file_size(path.c_str());
    count_type utilSize = sizeof(T);
    assert_msg(((total_bytes / utilSize) == length), "(total_bytes / utilSize != length) In Function load_binFile(...)");

    size_t bytes_to_read = total_bytes; // Shared Memory need all read
    size_t read_offset = 0;             // strat from the head of graph file

    // Start reading data - Open the file and locate the read location
    int fin = open(path.c_str(), O_RDONLY); // O_RDONLY: Open the file in read-only mode
    assert_msg(lseek(fin, read_offset, SEEK_SET) == read_offset, "Read error In Function load_binFile(...)");

    T *array_ = new T[length];

    // Start reading data - The variables necessary to read the file
    size_t read_bytes = 0; // Number of bytes read by the current machine
    size_t offset = 0;
    T *array_temp = new T[chunk_size];
    while (read_bytes < bytes_to_read)
    {
        int64_t curr_read_bytes;

        if (bytes_to_read - read_bytes > utilSize * chunk_size)
        { // A chunk_size can store up to 2^20 edges
            curr_read_bytes = read(fin, array_temp, utilSize * chunk_size);
        }
        else
        {
            curr_read_bytes = read(fin, array_temp, bytes_to_read - read_bytes);
        }
        assert_msg_clear(curr_read_bytes >= 0, "Read error In Function load_binFile(...)");
        read_bytes += curr_read_bytes;
        countl_type curr_read = curr_read_bytes / utilSize; // Number of edges that have been read

        //> Processing logic
        // TODO ???? memcpy是不是更好些
        // memcpy(array_+offset, array_temp, sizeof(T) * curr_read);
        omp_parallel_for(countl_type util = 0; util < curr_read; util++) { array_[util + offset] = array_temp[util]; }
        offset += curr_read;
    }

    delete[] array_temp;
    assert_msg_clear(close(fin) == 0, "[load_binArray] Close Error!");
    Msg_info("Load_binArray:[%s] Finished !", basename((char *)(path.c_str())));

    return array_;

clear:
    if (array_temp != nullptr)
        delete[] array_temp;
    if (array_ != nullptr)
        delete[] array_;

    STOP;
}

/* ***************************************************************************************
 * Func: Save The Array To Graph File
 *
 * @param [std::string path]       The File To Save Array
 * @param [T* array]               The Save Array
 * @param [uint64_t length]        The Length Of Array
 * ***************************************************************************************/
template <typename T> void save_binArray(std::string outFile, T *array, const uint64_t length)
{
    std::ofstream out_file;
    out_file.open(outFile.c_str(),
                  std::ios_base::out | std::ios_base::binary); // Opens as a binary read and writes to disk
    if (!out_file.good())
        assert_msg(false, "Error opening out-file: %s", outFile.c_str());

#if (USE_BAR_TQDM == 1)
    tqdm *bar;
    if (length > BAR_THRESHOLD)
    {
        bar = new tqdm();
        // bar->set_theme_circle();
    }

#endif

    timer t;
    for (size_t i = 0; i < length; i++)
    {
        out_file.write((char *)(&array[i]), sizeof(T));

#if (USE_BAR_TQDM == 1)
        if (length > BAR_THRESHOLD)
            bar->progress(i, length, "[save_binArray]: ");
#endif

        // if ((i != 0) && (i % RATE_UTIL == 0))
        //     Msg_rate("Save Bin Array To File Has Finished %.2lf %%", static_cast<double>(i) / static_cast<double>(length) * 100);
    }
#if (USE_BAR_TQDM == 1)
    if (length > BAR_THRESHOLD)
        bar->finish();
#endif
    Msg_finish("save_binArray [%s] Has Finished, Used Time: %.2lf (ms)", basename((char *)(outFile.c_str())), t.get_time_ms());
    // #if (USE_BAR_TQDM == 1)
    //     printf("\n");
    // #endif
    out_file.close();
}

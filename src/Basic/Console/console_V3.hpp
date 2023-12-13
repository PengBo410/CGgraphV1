#pragma once

// 只适用于Linux
#include "Basic/Log/log.hpp"
#include <iostream>
#include <libgen.h>
#include <omp.h>

#define TOTAL_DEBUG       // 不带颜色的Console打印
#define TOTAL_DEBUG_COLOR // 带颜色的Console打印

#define ISCONSOLE 1 // 是否输出到Console
#define ISLOG 1     // 是否输出到Log

#define CONSOLE_COLOR 1 // 带颜色的Console打印

#define GPU_THREAD_ID (threadIdx.x + blockIdx.x * blockDim.x)
#define serverId_static 0

/**
 *  31;40;1m" 是 ANSI 转义序列中控制文本颜色和属性的一种形式。它由以下三个部分组成：
 * "31"：设置前景色为红色
 * "40"：设置背景色为黑色
 * "1m"：设置文本加粗(即高亮),2m:启用暗淡文本;3m:启用斜体文本;4m:启用下划线文本;5m:启用闪烁文本;7m:交换背景色和前景色;8m:隐藏文本(即使其不可见)
 */
#define ESC_START "\033["
#define ESC_END "\033[0m"
#define COLOR_FATAL "31;40;5m" // 5:并启用闪烁的文本
#define COLOR_ALERT "31;40;1m"
#define COLOR_CRIT "31;40;1m"
#define COLOR_ERROR "31;48;1m"  // 31:是红色
#define COLOR_WARN "33;40;1m"   // 33:黄色
#define COLOR_NOTICE "34;40;1m" // 34:蓝色
#define COLOR_CHECK "32;48;5m"  // 32:是绿色 + 闪烁
#define COLOR_WRITE "36;48;1m"  // 36:淡蓝色
#define COLOR_FREES "32;48;1m"  // 32:绿色
#define COLOR_FINSH "32;48;1m"  // 32:绿色
#define COLOR_INFOS "37;48;1m"  // 37:白色  "37;40;1m"  48对应RGB{30,30,30}
#define COLOR_RATES "33;48;1m"  // 33:黄色
#define COLOR_LOG "35m"         // 35:紫色
#define COLOR_LOG_GPU "38;5;6m" // 35:浅蓝色

//> assert_msg + assert_msg_clear + STOP
#if (CONSOLE_COLOR == 1)
//^ 带颜色Console
#define assert_msg(condition, format, ...)                                                                                                           \
    if (true)                                                                                                                                        \
    {                                                                                                                                                \
        if (__builtin_expect(!(condition), 0))                                                                                                       \
        {                                                                                                                                            \
            fprintf(stderr, ESC_START COLOR_ERROR "<ERROR-%d>:" format " -> T[%u] [%s:%u 行]\n" ESC_END, serverId_static, ##__VA_ARGS__,             \
                    omp_get_thread_num(), basename((char *)(__FILE__)), __LINE__);                                                                   \
            fflush(stderr);                                                                                                                          \
            if constexpr (ISLOG == 1)                                                                                                                \
            {                                                                                                                                        \
                global_logFile().myFlush();                                                                                                          \
                global_logFile().myClose();                                                                                                          \
            }                                                                                                                                        \
            exit(1);                                                                                                                                 \
        }                                                                                                                                            \
    }

#define assert_msg_clear(condition, format, ...)                                                                                                     \
    if (true)                                                                                                                                        \
    {                                                                                                                                                \
        if (__builtin_expect(!(condition), 0))                                                                                                       \
        {                                                                                                                                            \
            fprintf(stderr, ESC_START COLOR_ERROR "<ERGO-%d>: " format " -> T[%u] [%s:%u 行]\n" ESC_END, serverId_static, ##__VA_ARGS__,             \
                    omp_get_thread_num(), basename((char *)__FILE__), __LINE__);                                                                     \
            fflush(stderr);                                                                                                                          \
            if constexpr (ISLOG == 1)                                                                                                                \
                global_logFile().myFlush();                                                                                                          \
            goto clear;                                                                                                                              \
        }                                                                                                                                            \
    }

#define STOP assert_msg(false, "Clear-Stop")

#else
//^ 不带颜色Console
#define assert_msg(condition, format, ...)                                                                                                           \
    if (true)                                                                                                                                        \
    {                                                                                                                                                \
        if (__builtin_expect(!(condition), 0))                                                                                                       \
        {                                                                                                                                            \
            fprintf(stderr, "<ERROR-%d>: " format " -> T[%u] {%s: %u行}\n", serverId_static, ##args, omp_get_thread_num(),                           \
                    basename((char *)__FILE__), __LINE__);                                                                                           \
            fflush(stderr);                                                                                                                          \
            if constexpr (ISLOG == 1)                                                                                                                \
                global_logFile().myFlush();                                                                                                          \
            exit(1);                                                                                                                                 \
        }                                                                                                                                            \
    }

#define assert_msg_clear(condition, format, ...)                                                                                                     \
    if (true)                                                                                                                                        \
    {                                                                                                                                                \
        if (__builtin_expect(!(condition), 0))                                                                                                       \
        {                                                                                                                                            \
            fprintf(stderr, "<ERGO-%d>: " format " -> T[%u] {%s: %u行}\n", serverId_static, ##args, omp_get_thread_num(),                            \
                    basename((char *)__FILE__), __LINE__);                                                                                           \
            fflush(stderr);                                                                                                                          \
            if constexpr (ISLOG == 1)                                                                                                                \
                global_logFile().myFlush();                                                                                                          \
            goto clear;                                                                                                                              \
        }                                                                                                                                            \
    }

#define STOP assert_msg(false, "Clear-Stop")
#endif

//> Console + Log
#if (ISCONSOLE == 1) && (ISLOG == 1)

//^ 带颜色Console
#if (CONSOLE_COLOR == 1)
#define Msg_info(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_INFOS "[INFOS-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                   \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_finish(format, ...)                                                                                                                      \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_FINSH "[FINSH-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                   \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_check(format, ...)                                                                                                                       \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_CHECK "[CHECK-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                   \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_write(format, ...)                                                                                                                       \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_WRITE "[WRITE-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                   \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_free(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_FREES "[FREES-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                   \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_logs(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_LOG "[LOGER-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                     \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_major(format, ...)                                                                                                                       \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_NOTICE "[MAJOR-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                  \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_warn(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_WARN "[WARNS-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                    \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_error(format, ...)                                                                                                                       \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_ERROR "[ERRNB-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                   \
                basename((char *)__FILE__), __LINE__);                                                                                               \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_node(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        if (serverId_static == 0)                                                                                                                    \
        {                                                                                                                                            \
            fprintf(stderr, ESC_START COLOR_INFOS "[SERVE-0]: " format " -> [%s:%u 行]" ESC_END "\n", ##__VA_ARGS__, basename((char *)__FILE__),     \
                    __LINE__);                                                                                                                       \
        }                                                                                                                                            \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define GPU_info(format, ...)                                                                                                                        \
    printf(ESC_START COLOR_LOG_GPU "[GTId-%2d]: " format " -> [%s:%u L]" ESC_END "\n", GPU_THREAD_ID, ##__VA_ARGS__, __FILE__, __LINE__);

//^ 不带颜色Console
#else
#define Msg_info(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, "[INFOS-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_finish(format, ...)                                                                                                                      \
    {                                                                                                                                                \
        fprintf(stderr, "[FINSH-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_check(format, ...)                                                                                                                       \
    {                                                                                                                                                \
        fprintf(stderr, "[CHECK-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_write(format, ...)                                                                                                                       \
    {                                                                                                                                                \
        fprintf(stderr, "[WRITE-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_free(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, "[FREES-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_logs(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, "[LOGER-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_major(format, ...)                                                                                                                       \
    {                                                                                                                                                \
        fprintf(stderr, "[MAJOR-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_warn(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, "[WARNS-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_error(format, ...)                                                                                                                       \
    {                                                                                                                                                \
        fprintf(stderr, "[ERRNB-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);             \
        global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                    \
    }

#define Msg_node(format, ...)                                                                                                                        \
    {                                                                                                                                                \
        if (serverId_static == 0)                                                                                                                    \
        {                                                                                                                                            \
            fprintf(stderr, "[SERVE-0]: " format " -> [%s:%u 行]\n", ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);                           \
            global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);                \
        }                                                                                                                                            \
    }

#define GPU_info(format, ...)                                                                                                                        \
    printf("[GTId-%2d]: " format " -> [%s:%u L]"                                                                                                     \
           "\n",                                                                                                                                     \
           GPU_THREAD_ID, ##__VA_ARGS__, __FILE__, __LINE__);

#endif

//> Console
#elif (ISCONSOLE == 1) && (ISLOG == 0)

//^ 带颜色Console
#if (CONSOLE_COLOR == 1)

#define Msg_info(format, ...)                                                                                                                        \
    fprintf(stderr, ESC_START COLOR_INFOS "[INFOS-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                       \
            basename((char *)__FILE__), __LINE__)
#define Msg_finish(format, ...)                                                                                                                      \
    fprintf(stderr, ESC_START COLOR_FINSH "[FINSH-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                       \
            basename((char *)__FILE__), __LINE__)
#define Msg_check(format, ...)                                                                                                                       \
    fprintf(stderr, ESC_START COLOR_CHECK "[CHECK-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                       \
            basename((char *)__FILE__), __LINE__)
#define Msg_write(format, ...)                                                                                                                       \
    fprintf(stderr, ESC_START COLOR_WRITE "[WRITE-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                       \
            basename((char *)__FILE__), __LINE__)
#define Msg_free(format, ...)                                                                                                                        \
    fprintf(stderr, ESC_START COLOR_FREES "[FREES-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                       \
            basename((char *)__FILE__), __LINE__)
#define Msg_logs(format, ...)                                                                                                                        \
    fprintf(stderr, ESC_START COLOR_LOG "[LOGER-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                         \
            basename((char *)__FILE__), __LINE__)
#define Msg_major(format, ...)                                                                                                                       \
    fprintf(stderr, ESC_START COLOR_NOTICE "[MAJOR-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                      \
            basename((char *)__FILE__), __LINE__)
#define Msg_warn(format, ...)                                                                                                                        \
    fprintf(stderr, ESC_START COLOR_WARN "[WARNS-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                        \
            basename((char *)__FILE__), __LINE__, basename((char *)__FILE__), __LINE__)
#define Msg_error(format, ...)                                                                                                                       \
    fprintf(stderr, ESC_START COLOR_ERROR "[ERRNB-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##__VA_ARGS__,                       \
            basename((char *)__FILE__), __LINE__)
#define Msg_node(format, ...)                                                                                                                        \
    if (serverId_static == 0)                                                                                                                        \
    {                                                                                                                                                \
        fprintf(stderr, ESC_START COLOR_INFOS "[SERVE-0]: " format " -> [%s:%u 行]" ESC_END "\n", ##__VA_ARGS__, basename((char *)__FILE__),         \
                __LINE__);                                                                                                                           \
    }
#define GPU_info(format, ...)                                                                                                                        \
    printf(ESC_START COLOR_LOG_GPU "[GTId-%2d]: " format " -> [%s:%u L]" ESC_END "\n", GPU_THREAD_ID, ##__VA_ARGS__, __FILE__, __LINE__);

//^ 不带颜色Console
#else

#define Msg_info(format, ...)                                                                                                                        \
    fprintf(stderr, "[INFOS-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_finish(format, ...)                                                                                                                      \
    fprintf(stderr, "[FINSH-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_check(format, ...)                                                                                                                       \
    fprintf(stderr, "[CHECK-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_write(format, ...)                                                                                                                       \
    fprintf(stderr, "[WRITE-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_free(format, ...)                                                                                                                        \
    fprintf(stderr, "[FREES-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_logs(format, ...)                                                                                                                        \
    fprintf(stderr, "[LOGER-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_major(format, ...)                                                                                                                       \
    fprintf(stderr, "[MAJOR-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_warn(format, ...)                                                                                                                        \
    fprintf(stderr, "[WARNS-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_error(format, ...)                                                                                                                       \
    fprintf(stderr, "[ERRNB-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__), __LINE__)
#define Msg_node(format, ...)                                                                                                                        \
    if (serverId() == 0)                                                                                                                             \
    {                                                                                                                                                \
        fprintf(stderr, "[SERVE-0]: " format " -> {%s: %u行}\n", ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);                               \
    }
#define GPU_info(format, ...)                                                                                                                        \
    printf("[GTId-%2d]: " format " -> [%s:%u L]"                                                                                                     \
           "\n",                                                                                                                                     \
           GPU_THREAD_ID, ##__VA_ARGS__, __FILE__, __LINE__);

#endif

//> Log
#elif (ISCONSOLE == 0) && (ISLOG == 1)

#define Msg_info(format, ...)                                                                                                                        \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_finish(format, ...)                                                                                                                      \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_check(format, ...)                                                                                                                       \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_write(format, ...)                                                                                                                       \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_free(format, ...)                                                                                                                        \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_logs(format, ...)                                                                                                                        \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_major(format, ...)                                                                                                                       \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_warn(format, ...)                                                                                                                        \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_error(format, ...)                                                                                                                       \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);
#define Msg_node(format, ...)                                                                                                                        \
    global_logFile().log(serverId_static, omp_get_thread_num(), basename((char *)__FILE__), __LINE__, format, ##__VA_ARGS__);

//> -
#else

#define Msg_info(format, ...)
#define Msg_finish(format, ...)
#define Msg_check(format, ...)
#define Msg_write(format, ...)
#define Msg_free(format, ...)
#define Msg_logs(format, ...)
#define Msg_major(format, ...)
#define Msg_warn(format, ...)
#define Msg_error(format, ...)
#define Msg_node(format, ...)
#define GPU_info(format, ...)

#endif

/* ******************************************************************************[V3.0]***********************************************************************************************/
/*
#if defined(TOTAL_DEBUG_COLOR)

#define assert_msg(condition, format, args...) \
    if (true) { \
        if (__builtin_expect(!(condition), 0)) { \
            fprintf(stderr, ESC_START COLOR_ERROR "<ERROR-%d>:" format " -> T[%u] {%s: %u行}\n" ESC_END, serverId_static, ##args,
omp_get_thread_num(),
basename((char *)(__FILE__)), __LINE__);       \
            fflush(stderr); \
            exit(1); \
        } \
    }

#define assert_msg_clear(condition, format, args...) \
    if (true) { \
        if (__builtin_expect(!(condition), 0)) { \
            fprintf(stderr, ESC_START COLOR_ERROR "<ERGO-%d>: " format " -> T[%u] {%s: %u行}\n" ESC_END, serverId_static, ##args,
omp_get_thread_num(),
basename((char *)__FILE__), __LINE__);         \
            fflush(stderr); \
            goto clear; \
        } \
    }

#define Msg_info(format, ...) fprintf(stderr, ESC_START COLOR_INFOS "[INFOS-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static,
##__VA_ARGS__, basename((char *)__FILE__), __LINE__) #define Msg_finish(format, args...) fprintf(stderr, ESC_START COLOR_FINSH "[FINSH-%d]: " format "
-> [%s:%u 行]" ESC_END
"\n", serverId_static, ##args, basename((char *)__FILE__), __LINE__) #define Msg_check(format, args...) fprintf(stderr, ESC_START COLOR_CHECK
"[CHECK-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##args, basename((char *)__FILE__), __LINE__) #define Msg_write(format, args...)
fprintf(stderr, ESC_START COLOR_WRITE "[WRITE-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##args, basename((char *)__FILE__),
__LINE__) #define Msg_free(format, args...) fprintf(stderr, ESC_START COLOR_FREES "[FREES-%d]: " format " -> [%s:%u 行]" ESC_END "\n",
serverId_static, ##args, basename((char
*)__FILE__), __LINE__) #define Msg_logs(format, args...) fprintf(stderr, ESC_START COLOR_RATES "[LOGER-%d]: " format " -> [%s:%u 行]" ESC_END "\n",
serverId_static, ##args, basename((char *)__FILE__), __LINE__) #define Msg_major(format, args...) fprintf(stderr, ESC_START COLOR_NOTICE "[MAJOR-%d]:
" format "
-> [%s:%u 行]" ESC_END "\n", serverId_static, ##args, basename((char *)__FILE__), __LINE__)
#define Msg_warn(format, args...) \ fprintf(stderr, ESC_START COLOR_WARN "[WARNS-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##args,
basename((char *)__FILE__), __LINE__, basename((char *)__FILE__), __LINE__) #define Msg_error(format, args...) fprintf(stderr, ESC_START COLOR_ERROR
"[ERRNB-%d]: " format " -> [%s:%u 行]" ESC_END "\n", serverId_static, ##args, basename((char *)__FILE__), __LINE__)
#define Msg_node(format, args...) \
    if (serverId_static == 0) { \
        fprintf(stderr, ESC_START COLOR_INFOS "[SERVE-0]: " format " -> {%s: %u行}" ESC_END "\n", ##args, basename((char *)__FILE__), __LINE__); \
    }

#define STOP assert_msg(false, "Clear-Stop")

#elif defined(TOTAL_DEBUG)

#define assert_msg(condition, format, args...) \
    if (true) { \
        if (__builtin_expect(!(condition), 0)) { \
            fprintf(stderr, "<ERROR-%d>: " format " -> T[%u] {%s: %u行}\n", serverId_static, ##args, omp_get_thread_num(), basename((char *)__FILE__),
__LINE__);                                      \
            fflush(stderr); \
            exit(1); \
        } \
    }

#define assert_msg_clear(condition, format, args...) \
    if (true) { \
        if (__builtin_expect(!(condition), 0)) { \
            fprintf(stderr, "<ERGO-%d>: " format " -> T[%u] {%s: %u行}\n", serverId_static, ##args, omp_get_thread_num(), basename((char *)__FILE__),
__LINE__);
\
            fflush(stderr); \
            goto clear; \
        } \
    }

#define Msg_info(format, ...) fprintf(stderr, "[INFOS-%d]: " format " -> {%s: %u行}\n", serverId_static, ##__VA_ARGS__, basename((char *)__FILE__),
__LINE__) #define Msg_finish(format, args...) fprintf(stderr, "[FINSH-%d]: " format " -> {%s: %u行}\n", serverId_static, ##args, basename((char
*)__FILE__), __LINE__) #define Msg_check(format, args...) fprintf(stderr, "[CHECK-%d]: " format " -> {%s: %u行}\n", serverId_static, ##args,
basename((char *)__FILE__), __LINE__) #define Msg_write(format, args...) fprintf(stderr, "[WRITE-%d]: " format " -> {%s: %u行}\n", serverId_static,
##args, basename((char *)__FILE__), __LINE__) #define Msg_free(format, args...) fprintf(stderr, "[FREES-%d]: " format " -> {%s: %u行}\n",
serverId_static, ##args, basename((char *)__FILE__), __LINE__) #define Msg_logs(format, args...) fprintf(stderr, "[LOGER-%d]: " format " -> {%s:
%u行}\n", serverId_static, ##args, basename((char *)__FILE__), __LINE__) #define Msg_major(format, args...) fprintf(stderr, "[MAJOR-%d]: " format " ->
{%s: %u行}\n", serverId_static, ##args, basename((char *)__FILE__), __LINE__) #define Msg_warn(format, args...) fprintf(stderr, "[WARNS-%d]: " format
" -> {%s: %u行}\n", serverId_static, ##args, basename((char *)__FILE__), __LINE__) #define Msg_error(format, args...) fprintf(stderr, "[ERRNB-%d]: "
format " -> {%s: %u行}\n", serverId_static, ##args, basename((char *)__FILE__), __LINE__)
#define Msg_node(format, args...) \
    if (serverId() == 0) { \
        fprintf(stderr, "[SERVE-0]: " format " -> {%s: %u行}\n", ##args, basename((char *)__FILE__), __LINE__); \
    }

#define STOP assert_msg(false, "Clear-Stop")

#else

#define assert_msg(condition, format, args...) \
    if (true) { \
        if (__builtin_expect(!(condition), 0)) { \
            printf("【ERROR】:" format " -> T[%u] {%s: %u行}\n", ##args, omp_get_thread_num(), basename((char *)__FILE__), __LINE__); \
            exit(1); \
        } \
    }

#define assert_msg_clear(condition, format, args...) \
    if (true) { \
        if (__builtin_expect(!(condition), 0)) { \
            fprintf(stderr, "<ERGO-%d>: " format " -> T[%u] {%s: %u行}\n", serverId_static, ##args, omp_get_thread_num(), basename((char *)__FILE__),
__LINE__);
\
            fflush(stderr); \
            goto clear; \
        } \
    }

#define Msg_info(format, args...)
#define Msg_finish(format, args...)
#define Msg_check(format, args...)
#define Msg_write(format, args...)
#define Msg_free(format, args...)
#define Msg_logs(format, args...)
#define Msg_major(format, args...)
#define Msg_warn(format, args...)
#define Msg_error(format, args...)
#define Msg_node(format, args...)

#define STOP assert_msg(false, "Clear-Stop")
#endif
*/
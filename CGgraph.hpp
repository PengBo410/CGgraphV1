#pragma once
#include "Basic/Console/console_V3.hpp"
#include "Basic/Log/log.hpp"
#include <gflags/gflags.h>

void initEnv(int argc, char *argv[])
{
    //> gflags
    gflags::SetUsageMessage("Welcom To Access CGgraph:");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    //> LOG
    global_logFile().set_log_file();
}

void endEnv()
{
    //> LOG
    global_logFile().myFlush();
    global_logFile().myClose();

    Msg_node("The CGgraph Execution Completed");
}
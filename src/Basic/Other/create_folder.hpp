#pragma once

#include "Basic/Console/console_V3.hpp"
#include "unistd.h"

/*****************************************************************
 * 判断[文件夹]和[文件]是否存在
 *
 * @param 目标的路径
 *****************************************************************/
bool isExist(const char *obj)
{
    if (access(obj, F_OK) >= 0)
        return true;
    else
        return false;
}

void createFloder_recursive(const std::string &path)
{
    try
    {
        std::filesystem::create_directories(path);
    } catch (const std::exception &e)
    {
        Msg_error("Error creating directories [%s], error: %s", path.c_str(), e.what());
        exit(1);
    }
}

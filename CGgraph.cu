#include "Basic/Console/console_V3.hpp"
#include "CGgraph.hpp"
#include "project/main_enginer.hpp"
#include <iostream>
#include <omp.h>

int main(int argc, char *argv[])
{

    initEnv(argc, argv);

    main_enginer();

    endEnv();
    return 0;
}

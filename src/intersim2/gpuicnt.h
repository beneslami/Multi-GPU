//
// Created by Ben on 4/18/21.
//

#ifndef MULTI_GPU_GPUICNT_H
#define MULTI_GPU_GPUICNT_H

#include <time.h>
#include <fstream>
#include <iostream>

class InterGPU {
public:
    InterGPU();
    void apply(const char*);
    void apply2(const char*);
};

#endif

//
// Created by Ben on 1/30/21.
//

#ifndef MULTI_GPU_GPUICNT_H
#define MULTI_GPU_GPUICNT_H

#include <time.h>
#include <fstream>
#include <iostream>
#include "../gpgpu-sim/mem_fetch.h"

class InterGPU {
    protected:
        size_t packet_count;
        size_t packet_size;
        size_t packet_type;
    public:
        InterGPU();
        void apply(const char*, unsigned , mem_fetch*, unsigned long long);
};

#endif

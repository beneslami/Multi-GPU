//
// Created by Ben on 1/30/21.
//

#ifndef MULTI_GPU_GPUICNT_H
#define MULTI_GPU_GPUICNT_H

#include <time.h>
#include <fstream>
#include <iostream>

class InterGPU {
    protected:
        size_t packet_count;
        size_t packet_size;
        size_t packet_type;
    public:
        InterGPU();
        void apply(const char*, unsigned, unsigned, unsigned int, int, int, unsigned int, const char*, unsigned long long);
};

#endif

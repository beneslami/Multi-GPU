//
// Created by Ben on 1/30/21.
//

#ifndef MULTI_GPU_GPUICNT_H
#define MULTI_GPU_GPUICNT_H

#include <time.h>

class InterGPU {
    protected:
        clock_t start;
        clock_t end;
        size_t packet_count;
        size_t packet_size;
        size_t packet_type;
    public:
        InterGPU();
        void setStart();
        void setEnd();
        void apply(const char*, unsigned, unsigned, unsigned int, int, unsigned int, char*);
};

#endif

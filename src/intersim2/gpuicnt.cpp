//
// Created by Ben on 1/30/21.
//

#include "gpuicnt.h"

InterGPU::InterGPU() {
    std::ofstream file("remote.txt", file.out | file.app);
}

void InterGPU::apply() {
    file.open("remote.txt", std::ios::app);
    if(file.is_open()){
        file << this->end - this->start << std::endl;
    }
    file.close();
}

void InterGPU::setStart() {
    this->start = clock();
}

void InterGPU::setEnd() {
    this->end = clock();
}
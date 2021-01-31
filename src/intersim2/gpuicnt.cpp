//
// Created by Ben on 1/30/21.
//

#include "gpuicnt.h"

InterGPU::InterGPU() {
    std::ofstream file("remote.txt", file.out | file.app);
    file.open("remote.txt", std::ios::app);
    file << "input\t" << "output\t" << "size\t" << "chip ID\t" << "sub_part ID\t" << "read/write\t" << "cycle(s)\n";
    file.close();
}

void InterGPU::apply(unsigned input_deviceID, unsigned output_deviceID, unsigned int size, int chip_id, unsigned int sub_partition_id, char* is_write) {
    file.open("remote.txt", std::ios::app);
    if(file.is_open()){
        file << input_deviceID << "\t" << output_deviceID << "\t" << size << "\t" << chip_id << "\t" << sub_partition_id << "\t" << is_write << "\t" << this->end - this->start << std::endl;
    }
    file.close();
}

void InterGPU::setStart() {
    this->start = clock();
}

void InterGPU::setEnd() {
    this->end = clock();
}
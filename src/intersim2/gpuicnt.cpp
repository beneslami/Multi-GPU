//
// Created by Ben on 1/30/21.
//

#include "gpuicnt.h"
#include <ctime>
#include <cstring>
#include "trafficmanager.hpp"

std::fstream file;

InterGPU::InterGPU() {
    file.open("icnt.txt", std::ios::app);
    //file.close();
}

void InterGPU::apply(const char* str) {
    file.open("icnt.txt", std::ios::app);
    if(file.is_open()) {
        file << str;
    }
    file.close();

}
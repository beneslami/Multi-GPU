//
// Created by Ben on 1/30/21.
//

#include "gpuicnt.h"
#include <ctime>
#include <cstring>
#include "trafficmanager.hpp"

std::fstream file;

InterGPU::InterGPU() {
    //file.open("remote.txt", std::ios::app);
    //file << "\tSource\t" << "Destination\t" << "hop\t" << "Size\t" << "Packet_Type\t" << "cycle(s)\t" << "time\n";
    //file.close();
}

void InterGPU::apply(const char* func, unsigned next_hop, mem_fetch *mf, unsigned long long cycle) {
    char type[15];
    unsigned input_deviceID = mf->get_src();
    /*switch (ptype) {
        case 0:
            strcpy(type, "READ_REQUEST");
            break;
        case 1:
            strcpy(type, "WRITE_REQUEST");
            break;
        case 2:
            strcpy(type, "READ_REPLY");
            break;
        case 3:
            strcpy(type, "WRITE_ACK");
            break;
    }*/

    file.open("remote.txt", std::ios::app);
    time_t ttime = time(0);
    tm *current = localtime(&ttime);
    //if(file.is_open()){
    //    file << func << "\t" << input_deviceID << "\t" << output_deviceID << "\t" << size << "\t" << type << "\t" << chip_id << "\t\t" << sub_partition_id << "\t\t" << is_write << "\t\t" << cycle << "\t\t" << current->tm_min << ":" << current->tm_sec << std::endl;
    //}
    //file.close();
    switch(strcmp(func, "push")){
        case 0:
            std::cout << func <<"\t" << mf->get_packet_token() << "\t" << mf->get_src() << "\t" << mf->get_dst() << "\t" << next_hop << "\t" << mf->get_create()<< "\t" << mf->get_send() << cycle << std::endl;
            break;
        default:
            std::cout << func <<"\t" << mf->get_packet_token() << "\t" << mf->get_src() << "\t" << mf->get_dst() << "\t" << mf->get_receive() << "\t" << cycle << std::endl;
    }

}

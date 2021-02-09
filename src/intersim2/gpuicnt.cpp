//
// Created by Ben on 1/30/21.
//

#include "gpuicnt.h"
#include <ctime>
#include <cstring>
#include "trafficmanager.hpp"

std::fstream file;

InterGPU::InterGPU() {
    file.open("remote.txt", std::ios::app);
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
    if(file.is_open()) {
        switch (strcmp(func, "push")) {
            case 0:
                file << func << "\t" << mf->get_packet_token() << "\t" << mf->get_src() << "\t" << mf->get_dst()
                          << "\t" << next_hop << "\t" << type << "\t" << mf->size() << "\t" << mf->get_create() << "\t" << mf->get_send() << "\t" << cycle
                          << "\t"<< std::endl;
                break;
            default:
                file << func << "\t" << mf->get_packet_token() << "\t" << mf->get_src() << "\t" << mf->get_dst() << "\t" << type << "\t" << mf->size()
                        << "\t" << mf->get_receive() << "\t" << cycle << std::endl;
                break;
        }
    }
    file.close();

}

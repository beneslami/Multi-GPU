//
// Created by Ben on 2/20/21.
//

#ifndef MULTI_GPU_REPORT_H
#define MULTI_GPU_REPORT_H
#include <cstdio>
#include <fstream>

class Report{
private:
    static Report *instance;
    Report();
public:
    std::fstream ben_file;
    static Report *get_instance();
    void apply(const char*);
};
#endif //MULTI_GPU_REPORT_H

//
// Created by Ben on 2/20/21.
//

#ifndef MULTI_GPU_REPORT_H
#define MULTI_GPU_REPORT_H
#include <cstdio>
#include <fstream>

class Report{
    static Report *instance;
    int m_value;
    Report(int v = 0){
        m_value = v;
    }
public:
    std::fstream ben_file;
    static Report *get_instance(){
        if(!instance){
            instance = new Report;
        }
        return instance;
    }
    void apply(const char*);
};
#endif //MULTI_GPU_REPORT_H

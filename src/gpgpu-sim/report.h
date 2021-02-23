//
// Created by Ben on 2/20/21.
//

#ifndef MULTI_GPU_REPORT_H
#define MULTI_GPU_REPORT_H
#include <cstdio>
#include <fstream>

class report{
    static report *instance;
    int m_value;
    report(int v = 0){
        m_value = v;
    }
public:
    std::fstream ben_file;
    static report *get_instance(){
        if(instance == nullptr){
            instance = new report;
        }
        return instance;
    }
    void apply(const char*);
};
#endif //MULTI_GPU_REPORT_H

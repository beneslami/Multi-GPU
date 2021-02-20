//
// Created by Ben on 2/20/21.
//

#ifndef MULTI_GPU_REPORT_H
#define MULTI_GPU_REPORT_H
#include <cstdio>
#include <fstream>

class report{
private:
    static report *instance;
public:
    std::fstream ben_file;
    static report *get_instance(){
        if(!instance){
            instance = new report;
            return instance;
        }
    }
    report();
    void apply(const char*);
};
#endif //MULTI_GPU_REPORT_H

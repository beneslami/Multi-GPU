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
protected:
    Report() { }
    Report(Report const&){}
    Report& operator=(Report const&){}
public:
    std::fstream ben_file;
    std::fstream ben_file2;
    std::fstream ben_file3;
    static Report *get_instance();
    void apply(const char*);
    void apply2(const char*);
    void icnt_apply(const char*);
};
#endif //MULTI_GPU_REPORT_H
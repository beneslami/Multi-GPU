//
// Created by Ben on 5/5/22.
//

#ifndef MULTI_GPU_REPORT_H
#define MULTI_GPU_REPORT_H
#include <cstdio>
#include <fstream>

class Report{
private:
    static Report *instance;
protected:

    Report& operator=(Report const&){}
public:
    Report() { }
    Report(Report const&){}
    std::fstream ben_file;
    std::fstream ben_file2;
    std::fstream ben_file3;
    static Report *get_instance();
    void apply(const char*);
    void apply2(const char*);
    void icnt_apply(const char*);
};
#endif //MULTI_GPU_REPORT_H

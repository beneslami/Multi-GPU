//
// Created by Ben on 2/20/21.
//
#include "report.h"
Report* Report::instance = 0;

static Report *get_instance(){
    if(!instance){
        instance = new Report;
    }
    return instance;
}

void Report::apply(const char *str) {
    if(ben_file.is_open()){
        ben_file << str ;
    }
    else{
        ben_file.open("report.txt", std::ios::app);
        ben_file << str ;
    }
    ben_file.close();
}

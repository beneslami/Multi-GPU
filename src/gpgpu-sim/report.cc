//
// Created by Ben on 2/20/21.
//
#include "report.h"
Report *Report::instance = NULL;

Report *Report::get_instance(){
    if(!Report::instance){
        Report::instance = new Report();
    }
    return  Report::instance;
}
void Report::apply(std::ostringstream str) {
    if(ben_file.is_open()){
        ben_file << str.str() ;
    }
    else{
        ben_file.open("report.txt", std::ios::app);
        ben_file << str.str() ;
    }
    ben_file.close();
}

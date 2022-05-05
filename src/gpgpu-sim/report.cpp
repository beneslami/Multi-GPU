//
// Created by Ben on 5/5/22.
//

#include "report.h"
Report *Report::instance = NULL;

/*Report *Report::get_instance(){
    if(!Report::instance){
        Report::instance = new Report();
    }
    return  Report::instance;
}*/

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

void Report::apply2(const char *str){
    if(ben_file2.is_open()){
        ben_file2 << str ;
    }
    else{
        ben_file2.open("boundary.txt", std::ios::app| std::ios::binary);
        ben_file2 << str ;
    }
    ben_file2.close();
}

void Report::icnt_apply(const char *str){
    if(ben_file3.is_open()){
        ben_file3 << str ;
    }
    else{
        ben_file3.open("icnt.txt", std::ios::app| std::ios::binary);
        ben_file3 << str ;
    }
    ben_file3.close();
}

/*
This header file is created by Ben @beneslami. This header file contains
a class which counts the number of occurrence of a remote request per second.
 */

#include "RateCounter.h"
#include <iostream>
#include <cstdio>
#include <fstream>
#include <iomanip>

RateCount::RateCount(size_t period)
{
    std::ofstream file("remote.txt", file.out | file.app);
    m_lastFlush = std::chrono::high_resolution_clock::now();
    m_period = (period / double(CLOCKS_PER_SEC))*(1000000);
    m_count = 0;
}

void RateCount::count()
{
    m_count++;
    std::chrono::system_clock::time_point now = std::chrono::high_resolution_clock::now();
    long time_slot_microsecond = std::chrono::duration_cast<std::chrono::microseconds>(now - m_lastFlush).count();

    if(time_slot_microsecond >= m_period){
        file.open("remote.txt", std::ios::app);
        //size_t count_per_sec = 0.0;
        if(m_count > 0)
        {
            //count_per_sec = m_count / (now - m_lastFlush);
            file.precision(5);
            //file << m_count << " in " << time_slot_microsecond << " Microseconds"<< std::endl;
            file << time_slot_microsecond << std::endl;
        }
        //std::cout << count_per_sec << " remote access per second" << std::endl;
        m_count = 0;
        m_lastFlush = now;
        file.close();
    }
}

/*
This header file is created by Ben @beneslami. This header file contains
a class which counts the number of occurrence of a remote request per second.
 */

#include "RateCounter.h"
#include <iostream>
#include <fstream>

RateCount::RateCount(time_t period) :
                                    m_lastFlush(std::time(NULL)),
                                    m_period(period),
                                    m_count(0)
{
    std::ofstream file("../../remote.txt", file.out | file.app);
}

void RateCount::count()
{
    m_count++;
    time_t now = std::time(NULL);
    if((now - m_lastFlush) >= m_period)
    {
        file.open("../../remote.txt", std::ios::app);
        size_t count_per_sec = 0.0;
        if(m_count > 0)
        {
            count_per_sec = m_count / (now - m_lastFlush);
            file << count_per_sec << "-->" << now;
        }
        std::cout << count_per_sec << " remote access per second" << std::endl;
        m_count = 0;
        m_lastFlush = now;
    }
}

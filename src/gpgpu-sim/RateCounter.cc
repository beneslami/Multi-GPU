/*
This header file is created by Ben @beneslami. This header file contains
a class which counts the number of occurrence of a remote request per second.
 */

#include "RateCounter.h"

RateCount::RateCount(time_t period) : 
                                    m_lastFlush(std::time(NULL)),
                                    m_period(period),
                                    m_count(0)
{}

void RateCount::count()
{
    m_count++;
    time_t now = std::time(NULL);
    if{(now - m_lastFlush) >= m_period)
    {
        size_t count_per_sec = 0.0;
        if(m_count > 0)
        {
            count_per_sec = m_count / (now - m_lastFlush);
        }
        std::cout << count_per_sec << " remote access per second" << endl;
        m_count = 0;
        m_lastFlush = now;
    }
}

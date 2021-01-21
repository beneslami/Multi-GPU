/*
This header file is created by Ben @beneslami. This header file contains
a class which counts the number of occurrence of a remote request per second.
 */

#ifndef RATECOUNTER_H
#define RATECOUNTER_H

#include <ctime>
#include <fstream>
#include <chrono>

class RateCount
{
protected:
    std::chrono::system_clock::time_point m_lastFlush;
    time_t m_period;
    size_t m_count;
    std::ofstream file;
public:
    RateCount(size_t period);   // Constructor
    //~RateCount();               // Destructor
    void count();
};

#endif

/*
This header file is created by Ben @beneslami. This header file contains
a class which counts the number of occurrence of a remote request per second.
 */

#ifndef RATECOUNTER_H
#define RATECOUNTER_H

#include <ctime>
#include <fstream>

class RateCount
{
protected:
    time_t m_lastFlush;
    time_t m_period;
    time_t m_count;
    std::ofstream file;
public:
    RateCount(time_t period);   // Constructor
    //~RateCount();               // Destructor
    void count();
};

#endif

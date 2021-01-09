/*
This header file is created by Ben @beneslami. This header file contains
a class which counts the number of occurrence of a remote request per second.
 */

#ifndef RATECOUNTER_H
#define RATECOUNTER_H

#include <ctime>

class RateCount
{
protected:
    time_t m_lastFlush;
    time_t m_period;
    time_t m_count;

public:
    RateCount(time_t period);   // Constructor
    //~RateCount();               // Distructor 
    void count();
};

#endif

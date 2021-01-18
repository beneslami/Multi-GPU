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
    std::ofstream file("remote.txt", file.out | file.app);
}

void RateCount::count()
{
    m_count++;
    time_t now = std::time(NULL);
    tm *ltm = localtime(&now);
    if((now - m_lastFlush) >= m_period)
    {
        file.open("remote.txt", std::ios::app);
        size_t count_per_sec = 0.0;
        if(m_count > 0)
        {
            count_per_sec = m_count / (now - m_lastFlush);
            file << count_per_sec << " --> " << ltm->tm_mday << " " << ltm->tm_hour << ":" << ltm->tm_min << ":" << ltm->tm_sec << std::endl;

        }
        std::cout << count_per_sec << " remote access per second" << std::endl;
        m_count = 0;
        m_lastFlush = now;
        file.close();
    }
}

/*
 #include <iostream>
#include <ctime>
#include <sys/time.h>
#include <iomanip>
//#include <bits/stdc++.h>

class RateCount
{
protected:
    size_t m_count;
    clock_t m_lastFlush;
    size_t m_period;

public:
    RateCount(size_t period);   // Constructor
    //~RateCount();               // Distructor
    void count();
};

RateCount::RateCount(size_t period){
  m_lastFlush = clock();
  m_period = (period / double(CLOCKS_PER_SEC))*(1000000);
  m_count = 0;
}

void RateCount::count(){
  m_count++;
  clock_t now = clock();
  double time_slot_microsecond = ((now - m_lastFlush) / double(CLOCKS_PER_SEC))*(1000000); //

  if(time_slot_microsecond >= m_period){
    size_t count_rate = 0.0;
    if(m_count > 0){
        count_rate = m_count / m_period;
        std::cout << "first: " <<  m_lastFlush / double(CLOCKS_PER_SEC) << std::setprecision(10) << std::endl;
        std::cout << "last: " << now / double(CLOCKS_PER_SEC) << std::setprecision(10) << std::endl;
        std::cout << time_slot_microsecond << " : " << count_rate << std::endl;
        std::cout <<"----------" << std::endl;
    }
    //cout << time_slot_milisecond/std::chrono::milliseconds(1);
    m_count = 0;
    m_lastFlush = now;
  }
}

int main()
{
  RateCount rate_count = RateCount(1);
  for(long int i = 0; i < 1000000000; i++){
    if(i %4 == 0 ){
      rate_count.count();
    }
  }
  return 0;
}
 */

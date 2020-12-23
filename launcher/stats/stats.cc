#include "stats.h"

extern unsigned long long gpu_sim_cycle;
extern unsigned long long gpu_tot_sim_cycle;

unsigned long long Stats::get_curr_cycle()
{
  return gpu_tot_sim_cycle + gpu_sim_cycle;
}


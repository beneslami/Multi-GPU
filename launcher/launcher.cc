#include <iostream>

///////////////////////added by shiqing to use int getrusage(..)
#include <sys/time.h>
#include <sys/resource.h>

#include "simulator.h"
#include "launcher_option_parser.h"

static Simulator simulator;
extern void exit_simulation();
 
int
main(int argc, char * argv[])
{

///////////////////////added by shiqing
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  printf("ZSQ: getrusage()	ru_maxrss: %llu; ru_ixrss: %llu; ru_idrss: %llu\n", usage.ru_maxrss, usage.ru_ixrss, usage.ru_idrss);

  LauncherOptionParser opt(argc, argv);

  // initialize streams with number of child processes
  simulator.initialize_streams(opt.getNumberOfProcesses());
  simulator.initialize_scheduler(opt.getScheduler());

  while (1) {
    for (LauncherOptionParser::iterator it = opt.begin(), it_end = opt.end();
         it != it_end; ++it) {
      printf("GPGPU-Sim MK-Sim: Process %d => Already has a launched kernel? %s, Is first run done? %s\n", (*it)->getID(), (*it)->has_launched_kernel() ? "Yes" : "No", (*it)->is_first_run_done() ? "Yes" : "No");
      if (!(*it)->has_launched_kernel()) {
        if (!(*it)->is_first_run_done()) {
          simulator.get_ready_for_launch_or_terminate(*it);
          if ((*it)->is_first_run_done()) {
            simulator.launch_bogus_kernel(*it);
          }
        } else {
          simulator.launch_bogus_kernel(*it);
        }
      }
    }

    if (opt.is_first_run_done()) {
      getrusage(RUSAGE_SELF, &usage);
      printf("ZSQ: getrusage()	ru_maxrss: %llu; ru_ixrss: %llu; ru_idrss: %llu\n", usage.ru_maxrss, usage.ru_ixrss, usage.ru_idrss);
      break;
    }

    if (simulator.launch(&opt)) {
      // temporarily works for single kernel case
      opt.getScheduler()->print_statistics();
      opt.getScheduler()->clear_statistics();
    } else {
      // simulation is done
      opt.getScheduler()->print_statistics();
      opt.getScheduler()->clear_statistics();
      getrusage(RUSAGE_SELF, &usage);
      printf("ZSQ: getrusage()	ru_maxrss: %llu; ru_ixrss: %llu; ru_idrss: %llu\n", usage.ru_maxrss, usage.ru_ixrss, usage.ru_idrss);
      break;
    }
  }

  opt.print_wrapup();
///////////////////////added by shiqing
  getrusage(RUSAGE_SELF, &usage);
  printf("ZSQ: getrusage()	ru_maxrss: %llu; ru_ixrss: %llu; ru_idrss: %llu\n", usage.ru_maxrss, usage.ru_ixrss, usage.ru_idrss);
  
  exit_simulation();
}

#include "../src/cuda-sim/cuda-sim.h"

void
ptxinfo_addinfo()
{
  if( !strcmp("__cuda_dummy_entry__",get_ptxinfo_kname()) ) {
    // this string produced by ptxas for empty ptx files (e.g., bandwidth test)
    clear_ptxinfo();
    return;
  }
  Simulator::CUctx_st *context = simulator.GPGPUSim_Context();
  print_ptxinfo();
  context->add_ptxinfo( get_ptxinfo_kname(), get_ptxinfo_kinfo() );
  clear_ptxinfo();
}

void
register_ptx_function( const char *name, function_info *impl )
{
  // no longer need this
}


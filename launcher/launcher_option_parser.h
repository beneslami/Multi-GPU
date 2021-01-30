#ifndef __LAUNCHER_OPTION_PARSER_H__
#define __LAUNCHER_OPTION_PARSER_H__

#include <fstream>
#include <vector>

#include "mk-sched/mk_scheduler.h"
#include "child_process.h"

class LauncherOptionParser {
public:
  typedef std::vector<ChildProcess*>::iterator        iterator;
  typedef std::vector<ChildProcess*>::const_iterator  const_iterator;

  iterator begin()              { return childs.begin(); }
  iterator end()                { return childs.end();   }
  const_iterator begin()  const { return childs.begin(); }
  const_iterator end()    const { return childs.end();   }

public:
  LauncherOptionParser(int argc, char *argv[]);
  ~LauncherOptionParser() {
    if (scheduler) {
      delete scheduler;
    }
  }

  unsigned getNumberOfProcesses() const { return childs.size(); }
  MKScheduler* getScheduler() { return scheduler; }

public:
  bool is_first_run_done() const
  {
    bool result = true;
    for (const_iterator it = begin(), it_end = end();
         it != it_end; ++it) {
      result &= (*it)->is_first_run_done();
    }
    return result;
  }

  bool run_until_end() const    { return stopCond == RUN_UNTIL_END; }
  bool run_until_cycle() const  { return stopCond == RUN_UNTIL_CYCLE; }
  bool run_until_inst() const   { return stopCond == RUN_UNTIL_INST; }
  unsigned long long run_until() const { return stopCondValue; }
  bool has_run_until_inst_finished() const;
  void KAIN_reset_cycle(unsigned long long value)
  {
 		assert(stopCond == RUN_UNTIL_CYCLE); 
 		stopCondValue = value; 
  }

public:
  void print_wrapup() const;

private:
  // prints out how to use the launcher
  // after printing it will exit the program
  // input is the binary name
  void usage(const char* binaryname) const;

  void parse_args(int argc, char *argv[]);
  void parse_config();
  bool parse_option(std::string line, std::string option_name, std::vector<std::string>& value);

  std::ifstream launcherConfigFile;
  MKScheduler* scheduler;

  std::vector<ChildProcess*> childs;

  // The simulation stop condition
  enum StopSimulationCondition {
    RUN_UNTIL_END,
    RUN_UNTIL_CYCLE,
    RUN_UNTIL_INST,
    RUN_UNTIL_CTA,
    STOP_SIMULATION_CONDITION_END
  };

  StopSimulationCondition stopCond;
  unsigned long long stopCondValue;

  void set_run_condition(enum StopSimulationCondition cond, unsigned long long value)
  {
    stopCond = cond;
    stopCondValue = value;
  }
};

#endif // __LAUNCHER_OPTION_PARSER_H__


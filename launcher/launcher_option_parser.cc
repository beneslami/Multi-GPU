#include <iostream>
#include <string>
#include <cassert>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>

#include "launcher_option_parser.h"

// this is from "gpgpusim_entrypoint.c"
extern char *sg_argv[];
extern unsigned long long get_curr_cycle();

// this will be used by kernel_info_t.cc
unsigned long long LATENCY_LIMIT = 10500; // 15us

// whitespace
static const std::string WHITESPACE_CHARS(" \t\r\n");

LauncherOptionParser::LauncherOptionParser(int argc, char *argv[])
  : stopCond(RUN_UNTIL_END), stopCondValue(0)
{
  scheduler = NULL;
  // parse command line arguments
  parse_args(argc, argv);

  // parse launcher config file
  parse_config();

  // close launcher config file
  launcherConfigFile.close();
}

void
LauncherOptionParser::usage(const char* binaryname) const {
  std::cout << std::endl;
  std::cout << "Usage: " << binaryname << " -l [launcher config file] -g [gpgpu-sim config file]" << std::endl;
  std::cout << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -l [launcher config file]" << std::endl;
  std::cout << "      Specifies the configuration for the launcher" << std::endl;
  std::cout << "  -g [gpgpu-sim config file]" << std::endl;
  std::cout << "      Specifies GPGPU architecture" << std::endl;
  std::cout << std::endl;
  exit(0);
}

void
LauncherOptionParser::parse_args(int argc, char *argv[]) {
  if (argc == 1) {
    usage(argv[0]);
  }

  static struct option long_options[] = {
    {"launcher-config", required_argument, 0, 'l'},
    {"gpgpu-config", required_argument, 0, 'g'},
    {0, 0, 0, 0}
  };

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "l:g:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'l':
        launcherConfigFile.open(optarg);
        if (!launcherConfigFile.is_open()) {
          std::cout << "Unable to open launcher config file: " << optarg << std::endl;
          exit(0);
        }
        break;

      case 'g':
        // overwrites the gpgpusim.config filename
        sg_argv[2] = strdup(optarg);
        break;

      default:
        usage(argv[0]);
    }
  }
}

bool Stream1_SM[384] = {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false, false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false};
bool Stream2_SM[192] = {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false};
bool Stream3_SM[192] = {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false};
bool Stream4_SM[192] = {false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false,false};


void
LauncherOptionParser::parse_config() {
  std::string line;
  int current_line_number = 0;

  //loop through the entire file
  while (launcherConfigFile) {
    std::getline(launcherConfigFile, line);
    current_line_number++;

    // skip empty lines
    if (line.size() == 0) {
      continue;
    }

    // find the first non-whitespace character
    std::string::size_type start = line.find_first_not_of(WHITESPACE_CHARS);

    // skip comment
    if (line.c_str()[start] == '#') {
      continue;
    }

    // skip whitespace lines
    if (start == std::string::npos) {
      continue;
    }

    std::string line_stripped = line.substr(start);
    std::vector<std::string> option_value;
    if (parse_option(line_stripped, "Cycle", option_value)) {
      assert(run_until_end());
      assert(option_value.size() == 1);
      set_run_condition(RUN_UNTIL_CYCLE, std::stoull(option_value[0]));
      continue;
    }

    if (parse_option(line_stripped, "Inst", option_value)) {
      assert(run_until_end());
      assert(option_value.size() == 1);
      set_run_condition(RUN_UNTIL_INST, std::stoull(option_value[0]));
      continue;
    }

    if (parse_option(line_stripped, "Stream1", option_value)) {

      for(int i = 0; i < option_value.size(); i++)
      {
            Stream1_SM[atoi(option_value[i].c_str())] = true;
      }
      continue;
    }

    if (parse_option(line_stripped, "Stream2", option_value)) {
      for(int i = 0; i < option_value.size(); i++)
      {
            Stream2_SM[atoi(option_value[i].c_str())] = true;
      }
      continue;
    }

    if (parse_option(line_stripped, "Stream3", option_value)) {
      for(int i = 0; i < option_value.size(); i++)
      {
            Stream3_SM[atoi(option_value[i].c_str())] = true;
      }
      continue;
    }

    if (parse_option(line_stripped, "Stream4", option_value)) {
      for(int i = 0; i < option_value.size(); i++)
      {
            Stream4_SM[atoi(option_value[i].c_str())] = true;
      }
      continue;
    }


    if (parse_option(line_stripped, "Block", option_value)) {
      assert(run_until_end());
      set_run_condition(RUN_UNTIL_CTA, std::stoull(option_value[0]));
      continue;
    }

    if (parse_option(line_stripped, "SchedulerType", option_value)) {
      SchedulerInfo info;
      info.add("SchedulerType", option_value);
      scheduler = MKScheduler::Create(info);
      continue;
    }

    if (parse_option(line_stripped, "PreemptionLatency", option_value)) {
      assert(LATENCY_LIMIT == 10500);
      LATENCY_LIMIT = std::stoull(option_value[0]);
      continue;
    }

    // otherwise, execution command
    // get the first item, binary name
    std::string::size_type end = line_stripped.find_first_of(WHITESPACE_CHARS);
    std::string binary = line_stripped.substr(0, end);

    std::vector<std::string> args;
    while (end != std::string::npos) {
      start = line_stripped.find_first_not_of(WHITESPACE_CHARS, end);
      if (start == std::string::npos) {
        // trailing whitespaces
        break;
      }

      end = line_stripped.find_first_of(WHITESPACE_CHARS, start);
      args.push_back(line_stripped.substr(start, end-start));
    }

    childs.push_back(new ChildProcess(binary, args));
  }

  int j = 0;//must set Stream1, Stream2, and the sum should be 16, the total number of SMs
  for(int i = 0; i < 24; i++)
  {
     if(Stream1_SM[i] == true && Stream2_SM[i] == true)
     {   
            assert(0);
     }   
     if(Stream1_SM[i] == true && Stream3_SM[i] == true)
     {   
            assert(0);
     }   
     if(Stream1_SM[i] == true && Stream4_SM[i] == true)
     {   
            assert(0);
     }   
     if(Stream2_SM[i] == true && Stream3_SM[i] == true)
     {   
            assert(0);
     }   
     if(Stream2_SM[i] == true && Stream4_SM[i] == true)
     {   
            assert(0);
     }   
     if(Stream3_SM[i] == true && Stream4_SM[i] == true)
     {   
            assert(0);
     }   
     if(Stream1_SM[i] == true || Stream2_SM[i] == true || Stream3_SM[i] == true || Stream4_SM[i] == true)
        j++;
  }
  printf("j is %d\n",j);
  fflush(stdout);
  //assert(j == 24);



}

bool
LauncherOptionParser::parse_option(std::string line, std::string option_name, std::vector<std::string>& value) {
  // see if the option matches, case insensitive
  if( strncasecmp(line.c_str(), option_name.c_str(), option_name.length()) != 0 ) {
    return false;
  }

  // option matches
  //strip the leading option string
  std::string remaining_line = line.substr(option_name.length());
  //strip optional whitespace
  std::string::size_type start = remaining_line.find_first_not_of(WHITESPACE_CHARS);
  assert(start != std::string::npos);

  remaining_line = remaining_line.substr(start);
  //strip required '=' char
  assert(remaining_line[0] == '=');
  //strip optional whitespace
  std::string::size_type end = 1;
  while (end != std::string::npos) {
    start = remaining_line.find_first_not_of(WHITESPACE_CHARS, end);
    if (start == std::string::npos) {
      // trailing whitespaces
      break;
    }

    end = remaining_line.find_first_of(WHITESPACE_CHARS, start);
    value.push_back(remaining_line.substr(start, end-start));
  }
  return true;
}

bool
LauncherOptionParser::has_run_until_inst_finished() const
{
  assert(run_until_inst());
  bool result = true;
  for (const_iterator it = begin(), it_end = end(); it != it_end; ++it) {
    if ((*it)->get_num_simulated_insts() >= stopCondValue || (*it)->is_first_run_done()) {
      (*it)->reached_given_insts();
    } else {
      // if ((*it)->get_num_simulated_insts() < stopCondValue && !(*it)->is_first_run_done()) {
      result = false;
    }
  }
  // every process has proceeded more than the stopCondValue
  return result;
}

void
LauncherOptionParser::print_wrapup() const
{
  for (const_iterator it = begin(), it_end = end(); it != it_end; ++it) {
    printf("Simulated insts[%u] = %llu\n", (*it)->getID(), (*it)->get_num_simulated_insts());
    printf("Simulated cycles[%u] = %llu\n", (*it)->getID(), (*it)->get_num_simulated_cycles());
    printf("Wasted insts[%u] = %llu\n", (*it)->getID(), (*it)->get_num_wasted_insts());
  }
}


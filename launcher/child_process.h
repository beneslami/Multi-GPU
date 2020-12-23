#ifndef __CHILD_PROCESS_H__
#define __CHILD_PROCESS_H__

#include <sys/types.h>
#include <map>
#include <vector>
#include <string>

// these are from multikernel-sim
#include "communicate.h"
#include "../common/kernel_config.h"

class ChildProcess {
public:
  ChildProcess(const std::string& _binary, const std::vector<std::string>& _args)
    : binaryName(_binary), binaryArgs(_args),
      fat_binary_count(0),
      first_run_done(false),
      launched_kernel(false),
      no_more_stats(false),
      num_simulated_insts(0),
      num_simulated_cycles(0),
      num_wasted_insts(0)
  {
    ID = ChildProcess::nextID++;
    spawn();
  }

  // read message's header
  MULTIKERNEL_MESSAGES get_message_header()
  {
    MULTIKERNEL_MESSAGES msg;
    pipe->read(msg);
    return msg;
  }

  // get ID
  unsigned getID() const { return ID; }

  // get pipe
  Communicate* get_pipe()
  {
    return pipe;
  }

  // rewind to the beginning
  void rewind();

public:
  // host function pointers can be the same due to the virtual address
  // use this function to make it unique
  void* get_unique_pointer(void* p);
  // check if the first run of the kernel is done
  bool is_first_run_done() const { return first_run_done; }
  // check if it has launched a kernel
  bool has_launched_kernel() const { return launched_kernel; }
  // check if there is no remaining fat binary
  bool no_more_fat_binary() const { return fat_binary_count == 0; }
  unsigned get_num_fat_binary() const { return fat_binary_count; }

public:
  // modify fat_binary_count
  void inc_fat_binary()           { ++fat_binary_count; }
  void dec_fat_binary()           { --fat_binary_count; }
  // modify launched_kernel
  void set_launched_kernel()      { launched_kernel = true; }
  void unset_launched_kernel()    { launched_kernel = false; }
  // modify first_run_done
  void terminate()                { first_run_done = true; }

private:
  void spawn();

public:
  // hostVar is a pointer in the child process, which is used for a key
  // however, the string itself is also necessary
  // these methods find those mappings
  bool hasVarName(const void* hostVar) const { return cudaVarNames.find(hostVar) != cudaVarNames.end();  }
  void addVarName(const void* hostVar, char* realName);
  char* findVarName(const void* hostVar);

  // hostVar is a pointer in the child process, pointing to a textureReference, and used for a key
  // however, the content is also important
  bool hasTextureReference(const void* hostVar) const { return cudaTextureReferences.find(hostVar) != cudaTextureReferences.end(); }
  void addTextureReference(const void* hostVar, struct textureReference* texRef);
  struct textureReference* findTextureReference(const void* hostVar);

  void reached_given_insts() { no_more_stats = true; }

private:
  pid_t pid;
  Communicate* pipe;

  std::string binaryName;
  std::vector<std::string> binaryArgs;

private:
  // keeps track of const/global names
  std::map<const void*, char*> cudaVarNames;
  // keeps track of texture reference
  std::map<const void*, struct textureReference*> cudaTextureReferences;

private:
  static unsigned nextID;
  unsigned        ID;

  unsigned        fat_binary_count;

  bool first_run_done;
  bool launched_kernel;
  bool no_more_stats;

  /// keeps track of statistics for child process
private:
  unsigned long long num_simulated_insts;
  unsigned long long num_simulated_cycles;
  unsigned long long num_wasted_insts;

public:
  void inc_num_simulated_insts(unsigned active_count);
  void dec_num_simulated_insts(unsigned wasted_insts);
  unsigned long long get_num_simulated_insts() const  { return num_simulated_insts; }

  void inc_cycles();
  unsigned long long get_num_simulated_cycles() const  { return num_simulated_cycles; }

  unsigned long long get_num_wasted_insts() const  { return num_wasted_insts; }

public:
  struct kernel_build_info {
    const char* hostFun;
    kernel_config config;

    kernel_build_info(const char* _hostFun, const kernel_config& _config)
      : hostFun(_hostFun), config(_config)
    {}
  };

  void add_kernel(const char* hostFun, const kernel_config& config);
  kernel_build_info* get_next_bogus_kernel_and_rotate();

private:
  std::list<kernel_build_info*> rotating_kernel_list;
};

#endif // __CHILD_PROCESS_H__


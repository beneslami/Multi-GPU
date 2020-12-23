#include <iostream>
#include <cassert>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "child_process.h"

unsigned ChildProcess::nextID = 0;

void
ChildProcess::spawn() {
  pid = fork();
  if (pid < 0) {
    std::cout << "Failed to fork child process." << std::endl;
    exit(0);
  } else if (pid == 0) {
    //this is the child process
    const char **arg_list;
    arg_list = new const char*[binaryArgs.size()+2];

    arg_list[0] = binaryName.c_str();
    for(unsigned j = 1; j <= binaryArgs.size(); j++) {
      arg_list[j] = binaryArgs[j-1].c_str();
    }

    arg_list[binaryArgs.size() + 1] = NULL;
    execvp(binaryName.c_str(), (char* const *)arg_list);

    // FIXME: This error is not communicated to the parent
    std::cout << "Error executing child: " << binaryName << std::endl;
    std::cout << strerror(errno) << std::endl;
    exit(0);
  } else {
    // this is the parent process
    pipe = new Communicate(pid);
    assert(pipe != NULL);
  }
}

void
ChildProcess::rewind() {
  // anything we need to re-initialize
  delete pipe;

  // re-run
  spawn();
}

void*
ChildProcess::get_unique_pointer(void* p)
{
  // Assumptions
  // 64-bit pointers (48-bit virtual address)
  // MSB 16 bits are zero-ed, and child process's ID is prepended
  const unsigned long long MASK = (((unsigned long long)1) << 48) - 1;

  unsigned long long ptr = (unsigned long long)p;
  ptr &= MASK;
  ptr |= (((unsigned long long)ID) << 48);

  return (void*)ptr;
}

void
ChildProcess::addVarName(const void* hostVar, char* realName)
{
  assert(!hasVarName(hostVar));
  cudaVarNames[hostVar] = realName;
}

char*
ChildProcess::findVarName(const void* hostVar)
{
  assert(hasVarName(hostVar));
  return cudaVarNames[hostVar];
}

void
ChildProcess::addTextureReference(const void* hostVar, struct textureReference* texRef)
{
  assert(!hasTextureReference(hostVar));
  cudaTextureReferences[hostVar] = texRef;
}

struct textureReference*
ChildProcess::findTextureReference(const void* hostVar)
{
  assert(hasTextureReference(hostVar));
  return cudaTextureReferences[hostVar];
}

bool KAIN_stall_recording_kernel0 = false;
bool KAIN_stall_recording_kernel1 = false;

void
ChildProcess::inc_num_simulated_insts(unsigned active_count)
{
  if (!is_first_run_done() && !no_more_stats) {
//  if (1) {
    num_simulated_insts += active_count;
  }
  else
  {
  	;
  }
}

void
ChildProcess::dec_num_simulated_insts(unsigned wasted_insts)
{
  if (!is_first_run_done() && !no_more_stats) {
//  if (1) {
    assert(num_simulated_insts >= wasted_insts);
    num_simulated_insts -= wasted_insts;
    num_wasted_insts += wasted_insts;
  }
}

void
ChildProcess::inc_cycles()
{
  if (!is_first_run_done() && !no_more_stats) {
//  if (1) {
    ++num_simulated_cycles;
  }
}

void
ChildProcess::add_kernel(const char* hostFun, const kernel_config& config)
{
  rotating_kernel_list.push_back(new kernel_build_info(hostFun, config));
}

ChildProcess::kernel_build_info*
ChildProcess::get_next_bogus_kernel_and_rotate()
{
  kernel_build_info* next_kernel = rotating_kernel_list.front();
  rotating_kernel_list.pop_front();
  rotating_kernel_list.push_back(next_kernel);
  return next_kernel;
}


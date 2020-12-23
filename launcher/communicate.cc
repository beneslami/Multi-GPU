#include <iostream>
#include <sstream>
#include <cassert>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <fcntl.h>

#include "communicate.h"

const std::string Communicate::PARENT_TO_CHILD_PIPE_PREFIX = ".gpgpu-sim_ptoc_";
const std::string Communicate::CHILD_TO_PARENT_PIPE_PREFIX = ".gpgpu-sim_ctop_";
const int Communicate::MAX_WAIT_FOR_PIPE = 60;

/******************************************/
/* Top Communicate class implementation   */
/******************************************/

Communicate::Communicate() {
  parent = false;
  child_pid = getpid();

  initialize_for_child();
}

Communicate::Communicate(pid_t cpid) {
  parent = true;
  child_pid = cpid;

  initialize_for_parent();
}

Communicate::~Communicate() {
  if (!parent) {
    std::ostringstream oss_ptoc;
    std::ostringstream oss_ctop;

    oss_ptoc << PARENT_TO_CHILD_PIPE_PREFIX << child_pid;
    oss_ctop << CHILD_TO_PARENT_PIPE_PREFIX << child_pid;

    remove(oss_ptoc.str().c_str());
    remove(oss_ctop.str().c_str());
  }
}

void
Communicate::initialize_for_parent() {
  // create the name pipes, we append the child pid to the end to make it unique
  std::ostringstream oss_ptoc;
  oss_ptoc << PARENT_TO_CHILD_PIPE_PREFIX << child_pid;
  if(mkfifo(oss_ptoc.str().c_str(), S_IRUSR | S_IWUSR) != 0) {
    printf("Failed to create pipe: %s\n", oss_ptoc.str().c_str());
    // kill all child processes
    kill(child_pid, SIGTERM);
    exit(0);
  }

  std::ostringstream oss_ctop;
  oss_ctop << CHILD_TO_PARENT_PIPE_PREFIX << child_pid;
  if(mkfifo(oss_ctop.str().c_str(), S_IRUSR | S_IWUSR) != 0) {
    printf("Failed to create pipe: %s\n", oss_ctop.str().c_str());
    // kill all child processes
    kill(child_pid, SIGTERM);
    exit(0);
  }

  // open the write pipe first here, then the read pipe, do the opposite on the other process
  // note that this will block until child process connects
  // that is why we create the pipe and fork each process first, then open each pipe
  write_pipe = open(oss_ptoc.str().c_str(), O_WRONLY);
  if (write_pipe == -1) {
    printf("Failed to open pipe: %s\n", oss_ptoc.str().c_str());
    kill(child_pid, SIGTERM);
    exit(-1);
  }
  // mark for removal
  // it will only be removed when the both the child and parent closes the file
  remove(oss_ptoc.str().c_str());

  read_pipe = open(oss_ctop.str().c_str(), O_RDONLY);
  if (read_pipe == -1) {
    printf("Failed to open pipe: %s\n", oss_ctop.str().c_str());
    kill(child_pid, SIGTERM);
    exit(-1);
  }
//  printf("KAIN: parent: %d, pid %d, open read pipeline %d, open write pipeline %d\n",parent,getpid(), read_pipe,write_pipe);
  // mark for removal
  // it will only be removed when the both the child and parent closes the file
  remove(oss_ctop.str().c_str());
}

void
Communicate::initialize_for_child() {
  struct stat stFileInfo;

  std::ostringstream oss_ptoc;
  oss_ptoc << PARENT_TO_CHILD_PIPE_PREFIX << child_pid;
  // wait for the parent to create the named pipes
  int count = 0;
  while (stat(oss_ptoc.str().c_str(), &stFileInfo) != 0) {
    sleep(1);
    count++;
    if(count > MAX_WAIT_FOR_PIPE) {
      printf("Timed out waiting for pipe to be created.\n");
      exit(-1);
    }
  }

  std::ostringstream oss_ctop;
  oss_ctop << CHILD_TO_PARENT_PIPE_PREFIX << child_pid;
  // wait for file to be created
  count = 0;
  while (stat(oss_ctop.str().c_str(), &stFileInfo) != 0) {
    sleep(1);
    count++;
    if(count > MAX_WAIT_FOR_PIPE) {
      printf("Timed out waiting for pipe to be created.\n");
      exit(-1);
    }
  }

  // open the pipes
  // note this will block until another process connects
  read_pipe = open(oss_ptoc.str().c_str(), O_RDONLY);
  if(read_pipe == -1) {
    printf("GPGPU-Sim PTX: Initialization error: Unable to open pipe to parent process.\n");
    exit(-1);
  }

  // note this will block until another process connects
  write_pipe = open(oss_ctop.str().c_str(), O_WRONLY);
  if(write_pipe == -1) {
    printf("GPGPU-Sim PTX: Initialization error: Unable to open pipe to parent process.\n");
    exit(-1);
  }
}

void
Communicate::recv(void *data, size_t num_bytes) {

  size_t total_bytes_read = 0;
  size_t kain_num_bytes = num_bytes;
  while (total_bytes_read < kain_num_bytes) {
    ssize_t bytes_read = ::read(read_pipe, data, num_bytes);

    if(bytes_read == 0) {
        if (parent) {
        printf("Benchmark terminated without notifying GPGPU-Sim.\n");
        } else {
        printf("GPGPU-Sim terminated without notifying benchmark.\n");
        }   
        exit(-1);
    }   
    if (bytes_read < 0) {
        if(parent) {
        printf("Error reading pipe in GPGPU-Sim.\n");
        } else {
        printf("Error reading pipe in benchmark.\n");
        }   
        exit(-1);
    }   
    printf("bytes_read %lld, num_bytes %lld\n",bytes_read,num_bytes);
    fflush(stdout);
    total_bytes_read += bytes_read;
    num_bytes -= bytes_read;
    data = &((char*)data)[bytes_read];
  }
  printf("out read while\n");
  fflush(stdout);


//  printf("KAIN: parent: %d, come here read here pipeline %d, gott it\n",parent,read_pipe);
}

void
Communicate::send(const void *data, size_t num_bytes) {
  size_t total_bytes_sent = 0;
size_t kain_number_bytes = num_bytes;
  while (total_bytes_sent < kain_number_bytes) {
//  	printf("KAIN: parent: %d, come here write here pipeline %d\n",parent,write_pipe);
	fflush(stdout);
    ssize_t bytes_sent = ::write(write_pipe, data, num_bytes);
    if (bytes_sent < 0) {
      if (parent) {
        printf("Error writing pipe in GPGPU-Sim.\n");
      } else {
        printf("Error writing pipe in benchmark.\n");
      }
      exit(-1);
    }

    total_bytes_sent += bytes_sent;
    num_bytes -= bytes_sent;
    data = &((const char*)data)[bytes_sent];
  }
}

/******************************************/
/* CommunicateWithCUDART implementation   */
/******************************************/

/******************************************/
/* CommunicateWithGPGPUSim implementation */
/******************************************/


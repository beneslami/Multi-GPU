#ifndef __COMMUNICATE_H__
#define __COMMUNICATE_H__

#include <unistd.h>
#include <string>

//This defines the different types of messages that
//can be sent between the child and parent processes
enum MULTIKERNEL_MESSAGES {
  MESSAGE_GPU_MALLOC,
  MESSAGE_GPU_MALLOCARRAY,
  MESSAGE_GPU_MEMCPY,
  MESSAGE_GPU_MEMCPY_SYMBOL,
  MESSAGE_GPU_MEMCPY_2D_TO_ARRAY,
  MESSAGE_GPU_MEMSET,
  MESSAGE_GPU_GET_DEVICE_COUNT,
  MESSAGE_GPU_GET_DEVICE_PROPERTY,
  MESSAGE_GPU_SET_DEVICE,
  MESSAGE_GPU_GET_DEVICE,
  MESSAGE_GPU_BIND_TEXTURE,
  MESSAGE_GPU_BIND_TEXTURE_TO_ARRAY,
  MESSAGE_GPU_GET_LAST_ERROR,
  MESSAGE_GPU_CONFIGURE_CALL,
  MESSAGE_GPU_SETUP_ARGUMENT,
  MESSAGE_GPU_LAUNCH,
  MESSAGE_GPU_SYNCHRONIZE,
  MESSAGE_GPU_REGISTER_FAT_BINARY,
  MESSAGE_GPU_REGISTER_FUNCTION,
  MESSAGE_GPU_REGISTER_CONST_VARIABLE,
  MESSAGE_GPU_REGISTER_GLOBAL_VARIABLE,
  MESSAGE_GPU_REGISTER_TEXTURE,
  MESSAGE_GPU_SET_CACHE_CONFIG,
  MESSAGE_GPU_EXIT_SIMULATION,
  MESSAGE_GPU_CURR_CYCLE,
  MESSAGE_GPU_END
};

class Communicate {
public:
  // called by the child process
  Communicate();
  // called by the parent process
  Communicate(pid_t child_pid);
  ~Communicate();

public:
  // high level wrapper for reading/writing
  template<typename RECV_TYPE>
  void read(RECV_TYPE& msg);

  template<typename SEND_TYPE>
  void write(const SEND_TYPE& msg);

  void read(       void *data, size_t num_bytes) { recv(data, num_bytes); }
  void write(const void *data, size_t num_bytes) { send(data, num_bytes); }

private:
  // low level recv/send methods
  void recv(      void *data, size_t num_bytes);
  void send(const void *data, size_t num_bytes);

private:
  // creates the pipes and opens them
  void initialize_for_parent();
  void initialize_for_child();

private:
  static const std::string PARENT_TO_CHILD_PIPE_PREFIX;
  static const std::string CHILD_TO_PARENT_PIPE_PREFIX;
  static const int MAX_WAIT_FOR_PIPE;

  // true if this is used by parent process
  // false if this is used by child process
  bool parent;
  // child pid if used by parent process
  // pid of itself if used by child process
  pid_t child_pid;

  int write_pipe;
  int read_pipe;
};

template<typename RECV_TYPE>
void
Communicate::read(RECV_TYPE& msg)
{
  recv(&msg, sizeof(msg));
}

template<typename SEND_TYPE>
void
Communicate::write(const SEND_TYPE& msg)
{
  send(&msg, sizeof(msg));
}

#endif // __COMMUNICATE_H__


#ifndef __KERNEL_CONFIG_H__
#define __KERNEL_CONFIG_H__

#include "gpgpu_ptx_sim_arg.h"

#if !defined(__VECTOR_TYPES_H__) && !defined(__DIM3_DEFINED__)
struct dim3 {
   unsigned int x, y, z;
//   unsigned int cluster_id;
};
#define __DIM3_DEFINED__
#endif

class kernel_config {
public:
  kernel_config( dim3 GridDim, dim3 BlockDim, size_t sharedMem, struct CUstream_st *stream )
  {
    m_GridDim=GridDim;
    m_BlockDim=BlockDim;
    m_sharedMem=sharedMem;
    m_stream = stream;
  }
  void set_arg( const void *arg, size_t size, size_t offset )
  {
    m_args.push_front( gpgpu_ptx_sim_arg(arg,size,offset) );
  }
  dim3 grid_dim() const { return m_GridDim; }
  dim3 block_dim() const { return m_BlockDim; }
  gpgpu_ptx_sim_arg_list_t get_args() { return m_args; }
  struct CUstream_st *get_stream() { return m_stream; }

private:
  dim3 m_GridDim;
  dim3 m_BlockDim;
  size_t m_sharedMem;
  struct CUstream_st *m_stream;
  gpgpu_ptx_sim_arg_list_t m_args;
};

#endif // __KERNEL_CONFIG_H__


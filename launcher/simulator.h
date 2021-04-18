#ifndef __SIMULATOR_H__
#define __SIMULATOR_H__

#include <list>
#include <map>
#include <cassert>

#define __CUDA_RUNTIME_API_H__

// these are from NVIDIA SDK
#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "__cudaFatFormat.h"

// these are from GPGPU-sim
#include "../src/gpgpu-sim/gpu-sim.h"
#include "../src/cuda-sim/ptx_ir.h"

// these are from multikernel-sim
#include "../common/cuda_array.h"
#include "../common/kernel_config.h"

class ChildProcess;
class MKScheduler;
class LauncherOptionParser;

class Simulator {
public:
  /*DEVICE_BUILTIN*/
  // this is used only within the simulator that invokes GPGPU-sim
  struct _cuda_device_id {
    _cuda_device_id(gpgpu_sim* gpu) {m_id = 0; m_next = NULL; m_gpgpu=gpu;}
    struct _cuda_device_id *next() { return m_next; }
    unsigned num_shader() const { return m_gpgpu->get_config().num_shader(); }
    int num_devices() const {
      if( m_next == NULL ) return 1;
      else return 1 + m_next->num_devices();
    }
    struct _cuda_device_id *get_device( unsigned n ) {
      assert( n < (unsigned)num_devices() );
      struct _cuda_device_id *p=this;
      for(unsigned i=0; i<n; i++)
        p = p->m_next;
      return p;
    }
    const struct cudaDeviceProp *get_prop() const {
      return m_gpgpu->get_prop();
    }
    unsigned get_id() const { return m_id; }

    gpgpu_sim *get_gpgpu() { return m_gpgpu; }

  private:
    unsigned m_id;
    class gpgpu_sim *m_gpgpu;
    struct _cuda_device_id *m_next;
  };

  // this is used only within the simulator that invokes GPGPU-sim
  struct CUctx_st {
    CUctx_st( _cuda_device_id *gpu ) { m_gpu = gpu; }

    _cuda_device_id *get_device() { return m_gpu; }

    void add_binary( symbol_table *symtab, unsigned fat_cubin_handle ) {
      m_code[fat_cubin_handle] = symtab;
      m_last_fat_cubin_handle = fat_cubin_handle;
    }

    void add_ptxinfo( const char *deviceFun, const struct gpgpu_ptx_sim_kernel_info &info ) {
      symbol *s = m_code[m_last_fat_cubin_handle]->lookup(deviceFun);
      assert( s != NULL );
      function_info *f = s->get_pc();
      assert( f != NULL );
      f->set_kernel_info(info);
    }

    void register_function( unsigned fat_cubin_handle, const char *hostFun, const char *deviceFun ) {
      if( m_code.find(fat_cubin_handle) != m_code.end() ) {
        symbol *s = m_code[fat_cubin_handle]->lookup(deviceFun);
        assert( s != NULL );
        function_info *f = s->get_pc();
        assert( f != NULL );
        m_kernel_lookup[hostFun] = f;
      } else {
        m_kernel_lookup[hostFun] = NULL;
      }
    }

    function_info *get_kernel(const char *hostFun) {
      std::map<const void*,function_info*>::iterator i=m_kernel_lookup.find(hostFun);
      assert( i != m_kernel_lookup.end() );
      return i->second;
    }

  private:
    _cuda_device_id *m_gpu; // selected gpu
    std::map<unsigned,symbol_table*> m_code; // fat binary handle => global symbol table
    unsigned m_last_fat_cubin_handle;
    std::map<const void*,function_info*> m_kernel_lookup; // unique id (CUDA app function address) => kernel entry point
  };

public:
  Simulator();
  ~Simulator();

  // initialize streams such that each kernel
  // maps onto a stream (first kernel - stream zero, etc.)
  void initialize_streams(const unsigned numProcesses);
  // initialize multikernel scheduler for GPU
  void initialize_scheduler(MKScheduler* mk_sched);
  // process continues until there is a kernel launch or exit
  void get_ready_for_launch_or_terminate(ChildProcess* process);
  // the process has finished, launch bogus kernel from the beginning
  void launch_bogus_kernel(ChildProcess* process);
  // launch all the kernels waiting
  // returns true  if one of the kernel is finished
  // returns false if simulation was finished due to the script (cycle/inst)
  bool launch(LauncherOptionParser* opt);

private:
  // returns true if the message was cudaLaunch or exit
  // otherwise, returns false
  bool process_message(ChildProcess* process);

public:
  // initializations
  // they are called once at the first encounter of any CUDA API
  _cuda_device_id*  GPGPUSim_Init();
  CUctx_st*         GPGPUSim_Context();

  // CUDA APIs
  // Errors are no longer returned
  //// Memory allocation/deallocation
  void* cudaMalloc(size_t size);
  void* cudaMallocArray(size_t size);
  void cudaFree(void *devPtr);
  void cudaFreeArray(struct cudaArray *array);

  //// Memory copy
  void cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
  void cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
  void cudaMemcpy2DToArray(void *devPtr, const void *hostPtr, enum cudaMemcpyKind kind, size_t size);
  void cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset = 0);
  void cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset = 0);

  //// Memory copy - asynchronous
  void cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

  //// Memory set
  void cudaMemset(void *mem, int c, size_t count);

  //// Device API
  //// Currently, these are unused
  int cudaGetDeviceCount();
  const struct cudaDeviceProp* cudaGetDeviceProperties(int device);
  void cudaChooseDevice(const struct cudaDeviceProp *prop);
  void cudaSetDevice(int device);
  int cudaGetDevice();

  //// Texture API
  void cudaBindTexture(const struct textureReference* texref, const void *devPtr, const struct cudaChannelFormatDesc& desc, size_t size);
  void cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array);

  //// empty API
  void cudaGetLastError();

  //// Kernel launch
  void cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
  void cudaSetupArgument(const void *arg, size_t size, size_t offset);
  kernel_info_t* cudaLaunch( const char *hostFun, ChildProcess* parent );

  //// Thread management
  void cudaThreadSynchronize();

  //// PTX management
  void __cudaRegisterFatBinary(unsigned long long fat_cubin_handle, const char* ptx);
  void __cudaRegisterFunction(unsigned fat_cubin_handle, const char* hostFun, char* deviceFun);
  void __cudaRegisterTexture(const struct textureReference *hostVar, const char* deviceName, int dim, int norm, int ext);

  //// Stream management
  cudaStream_t cudaStreamCreate();
  void cudaStreamDestroy(cudaStream_t stream);

  //// GPU management
  void cudaFuncSetCacheConfig(const char * hostFunc, enum cudaFuncCache cacheConfig);

private:
  unsigned long long get_fat_cubin_handle();

private:
  _cuda_device_id*                            the_device;
  CUctx_st*                                   the_context;

  int                                         active_device;
  std::list<kernel_config>                    cuda_launch_stack;

  std::list<std::pair<ChildProcess*, const char*> > launch_kernels;
  // streams are mapped automatically for each process
  // therefore, use of streams within the process is forbidden
  std::vector<cudaStream_t>                   mapped_streams;
};

#endif // __SIMULATOR_H__


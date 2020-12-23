#include <stdarg.h>

#include "simulator.h"
#include "child_process.h"
#include "launcher_option_parser.h"
#include "mk-sched/mk_scheduler.h"

// these are from GPGPU-sim
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/stream_manager.h"
#include "../src/gpgpusim_entrypoint.h"

//
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __my_func__    __PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __my_func__    __func__
#  else
#   define __my_func__    ((__const char *) 0)
#  endif
# endif

// jasonjk begin
// define this for debugging
//#define DEBUG_MULTIKERNEL_SIM

#ifdef DEBUG_MULTIKERNEL_SIM
static FILE* debugFile = NULL;

#define PRINT_CALL { fprintf(debugFile, "%s\n", __my_func__); fflush(debugFile); }
#endif
// jasonjk end
extern void synchronize();
extern stream_manager *g_stream_manager;

/****************************************/
/* Helpers implementation               */
/* - visible only within this file      */
/****************************************/

static void
cuda_not_implemented(const char* func, unsigned line)
{
  fflush(stdout);
  fflush(stderr);
  printf("\n\nGPGPU-Sim PTX: Execution error: CUDA API function \"%s()\" has not been implemented yet.\n"
      "                 [libcuda/%s around line %u]\n\n\n",
      func, __FILE__, line);
  fflush(stdout);
  abort();
}

#define gpgpusim_ptx_error(msg, ...) gpgpusim_ptx_error_impl(__func__, __FILE__,__LINE__, msg, ##__VA_ARGS__)
#define gpgpusim_ptx_assert(cond,msg, ...) gpgpusim_ptx_assert_impl((cond),__func__, __FILE__,__LINE__, msg, ##__VA_ARGS__)

static void
gpgpusim_ptx_error_impl( const char *func, const char *file, unsigned line, const char *msg, ... )
{
  va_list ap;
  char buf[1024];
  va_start(ap,msg);
  vsnprintf(buf,1024,msg,ap);
  va_end(ap);

  printf("GPGPU-Sim CUDA API: %s\n", buf);
  printf("                    [%s:%u : %s]\n", file, line, func );
  abort();
}

static void
gpgpusim_ptx_assert_impl( int test_value, const char *func, const char *file, unsigned line, const char *msg, ... )
{
  va_list ap;
  char buf[1024];
  va_start(ap,msg);
  vsnprintf(buf,1024,msg,ap);
  va_end(ap);

  if ( test_value == 0 )
    gpgpusim_ptx_error_impl(func, file, line, msg);
}

static int
load_static_globals( symbol_table *symtab, unsigned min_gaddr, unsigned max_gaddr, gpgpu_t *gpu ) 
{
  printf( "GPGPU-Sim PTX: loading globals with explicit initializers... \n" );
  fflush(stdout);
  int ng_bytes=0;
  symbol_table::iterator g=symtab->global_iterator_begin();

  for ( ; g!=symtab->global_iterator_end(); g++) {
    symbol *global = *g;
    if ( global->has_initializer() ) {
      printf( "GPGPU-Sim PTX:     initializing '%s' ... ", global->name().c_str() );
      unsigned addr=global->get_address();
      const type_info *type = global->type();
      type_info_key ti=type->get_key();
      size_t size;
      int t;
      ti.type_decode(size,t);
      int nbytes = size/8;
      int offset=0;
      std::list<operand_info> init_list = global->get_initializer();
      for ( std::list<operand_info>::iterator i=init_list.begin(); i!=init_list.end(); i++ ) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        assert( (addr+offset+nbytes) < min_gaddr ); // min_gaddr is start of "heap" for cudaMalloc
        gpu->get_global_memory()->write(addr+offset,nbytes,&value,NULL,NULL); // assuming little endian here
        offset+=nbytes;
        ng_bytes+=nbytes;
      }
      printf(" wrote %u bytes\n", offset );
    }
  }
  printf( "GPGPU-Sim PTX: finished loading globals (%u bytes total).\n", ng_bytes );
  fflush(stdout);
  return ng_bytes;
}

static int
load_constants( symbol_table *symtab, addr_t min_gaddr, gpgpu_t *gpu ) 
{
  printf( "GPGPU-Sim PTX: loading constants with explicit initializers... " );
  fflush(stdout);
  int nc_bytes = 0;
  symbol_table::iterator g=symtab->const_iterator_begin();

  for ( ; g!=symtab->const_iterator_end(); g++) {
    symbol *constant = *g;
    if ( constant->is_const() && constant->has_initializer() ) {

      // get the constant element data size
      int basic_type;
      size_t num_bits;
      constant->type()->get_key().type_decode(num_bits,basic_type);

      std::list<operand_info> init_list = constant->get_initializer();
      int nbytes_written = 0;
      for ( std::list<operand_info>::iterator i=init_list.begin(); i!=init_list.end(); i++ ) {
        operand_info op = *i;
        ptx_reg_t value = op.get_literal_value();
        int nbytes = num_bits/8;
        switch ( op.get_type() ) {
          case int_t: assert(nbytes >= 1); break;
          case float_op_t: assert(nbytes == 4); break;
          case double_op_t: assert(nbytes >= 4); break; // account for double DEMOTING
          default:
                            abort();
        }
        unsigned addr=constant->get_address() + nbytes_written;
        assert( addr+nbytes < min_gaddr );

        gpu->get_global_memory()->write(addr,nbytes,&value,NULL,NULL); // assume little endian (so u8 is the first byte in u32)
        nc_bytes+=nbytes;
        nbytes_written += nbytes;
      }
    }
  }
  printf( " done.\n");
  fflush(stdout);
  return nc_bytes;
}

static kernel_info_t *gpgpu_cuda_ptx_sim_init_grid( const char *hostFun, 
		gpgpu_ptx_sim_arg_list_t args,
		struct dim3 gridDim,
		struct dim3 blockDim,
		Simulator::CUctx_st* context )
{
  function_info *entry = context->get_kernel(hostFun);
  kernel_info_t *result = new kernel_info_t(gridDim,blockDim,entry);
  if( entry == NULL ) {
    printf("GPGPU-Sim PTX: ERROR launching kernel -- no PTX implementation found for %p\n", hostFun);
    abort();
  }
  unsigned argcount=args.size();
  unsigned argn=1;
  for( gpgpu_ptx_sim_arg_list_t::iterator a = args.begin(); a != args.end(); a++ ) {
    entry->add_param_data(argcount-argn,&(*a));
    argn++;
  }

  entry->finalize(result->get_param_memory());
  g_ptx_kernel_count++;
  fflush(stdout);

  return result;
}

/****************************************/
/* Simulator implementation             */
/****************************************/
Simulator::Simulator()
  : the_device(NULL), the_context(NULL),
    active_device(0)
{
#ifdef DEBUG_MULTIKERNEL_SIM
  debugFile = fopen("debug.silver", "w");
#endif
}

Simulator::~Simulator()
{
  if (the_device) {
    delete the_device;
  }
  if (the_context) {
    delete the_context;
  }

#ifdef DEBUG_MULTIKERNEL_SIM
  fclose(debugFile);
#endif
}

Simulator::_cuda_device_id*
Simulator::GPGPUSim_Init()
{
  if (the_device == NULL) {
    gpgpu_sim *the_gpu = gpgpu_ptx_sim_init_perf();

    cudaDeviceProp *prop = (cudaDeviceProp *) calloc(sizeof(cudaDeviceProp),1);
    snprintf(prop->name,256,"GPGPU-Sim_v%s", g_gpgpusim_version_string );
    prop->major = 2;
    prop->minor = 0;
    prop->totalGlobalMem = 0x40000000 /* 1 GB */;
    prop->memPitch = 0;
    prop->maxThreadsPerBlock = 512;
    prop->maxThreadsDim[0] = 512;
    prop->maxThreadsDim[1] = 512;
    prop->maxThreadsDim[2] = 512;
    prop->maxGridSize[0] = 0x40000000;
    prop->maxGridSize[1] = 0x40000000;
    prop->maxGridSize[2] = 0x40000000;
    prop->totalConstMem = 0x40000000;
    prop->textureAlignment = 0;
    prop->sharedMemPerBlock = the_gpu->shared_mem_size();
    prop->regsPerBlock = the_gpu->num_registers_per_core();
    prop->warpSize = the_gpu->wrp_size();
    prop->clockRate = the_gpu->shader_clock();
#if (CUDART_VERSION >= 2010)
    prop->multiProcessorCount = the_gpu->get_config().num_shader();
#endif
    the_gpu->set_prop(prop);
    the_device = new _cuda_device_id(the_gpu);

    start_sim_thread(1);
  }

  return the_device;
}

Simulator::CUctx_st*
Simulator::GPGPUSim_Context()
{
  if (the_context == NULL) {
    _cuda_device_id* device = GPGPUSim_Init();
    the_context = new CUctx_st(device);
  }

  return the_context;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void*
Simulator::cudaMalloc(size_t size)
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif

  CUctx_st* context = GPGPUSim_Context();
  void* devPtr = context->get_device()->get_gpgpu()->gpu_malloc(size);
  if(g_debug_execution >= 3)
    printf("GPGPU-Sim PTX: cudaMallocing %zu bytes starting at 0x%llx..\n",size, (unsigned long long) devPtr);
  return devPtr;
}

void*
Simulator::cudaMallocArray(size_t size)
{
  CUctx_st* context = GPGPUSim_Context();
  void* devPtr = context->get_device()->get_gpgpu()->gpu_mallocarray(size);
  return devPtr;
}

void
Simulator::cudaFree(void *devPtr)
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif
  // TODO...  manage g_global_mem space?
  // FIXME: cudaFree is never called in multikernel-sim
  cuda_not_implemented(__my_func__,__LINE__);
}

void
Simulator::cudaFreeArray(struct cudaArray *array)
{
  // TODO...  manage g_global_mem space?
  // FIXME: cudaFree is never called in multikernel-sim
  cuda_not_implemented(__my_func__,__LINE__);
};

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif

  if(g_debug_execution >= 3)
    printf("GPGPU-Sim PTX: cudaMemcpy(): devPtr = %p\n", dst);

  if( kind == cudaMemcpyHostToDevice ) {
    g_stream_manager->push( stream_operation(src,(size_t)dst,count,0) );
#ifdef DEBUG_MULTIKERNEL_SIM
    /*
    const size_t converted_size = count / sizeof(float);
    for (size_t i = 0; i < converted_size; ++i) {
      fprintf(debugFile, "%f\n", ((const float*)src)[i]);
    }
    */
#endif
  } else if( kind == cudaMemcpyDeviceToHost ) {
    g_stream_manager->push( stream_operation((size_t)src,dst,count,0) );
  } else if( kind == cudaMemcpyDeviceToDevice ) {
    g_stream_manager->push( stream_operation((size_t)src,(size_t)dst,count,0) );
#ifdef DEBUG_MULTIKERNEL_SIM
    /*
    fprintf(debugFile, "device = %p, host = %p, count = %zu\n", src, dst, count);
    fflush(debugFile);
    const size_t converted_size = count / sizeof(float);
    for (size_t i = 0; i < converted_size; ++i) {
      fprintf(debugFile, "%f\n", ((const float*)dst)[i]);
    }
    fflush(debugFile);
    */
#endif

  } else {
    printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
}

void
Simulator::cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
  CUctx_st *context = GPGPUSim_Context();
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  size_t size = spitch*height;
  gpgpusim_ptx_assert( (dpitch==spitch), "different src and dst pitch not supported yet" );
  if( kind == cudaMemcpyHostToDevice )
    gpu->memcpy_to_gpu( (size_t)dst, src, size );
  else if( kind == cudaMemcpyDeviceToHost )
    gpu->memcpy_from_gpu( dst, (size_t)src, size );
  else if( kind == cudaMemcpyDeviceToDevice )
    gpu->memcpy_gpu_to_gpu( (size_t)dst, (size_t)src, size);
  else {
    printf("GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
}

void
Simulator::cudaMemcpy2DToArray(void *devPtr, const void *hostPtr, enum cudaMemcpyKind kind, size_t size)
{
  CUctx_st *context = GPGPUSim_Context();
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  if ( kind == cudaMemcpyHostToDevice ) {
    gpu->memcpy_to_gpu( (size_t)devPtr, hostPtr, size);
  } else if( kind == cudaMemcpyDeviceToHost ) {
    gpu->memcpy_from_gpu( devPtr, (size_t)hostPtr, size);
  } else if( kind == cudaMemcpyDeviceToDevice ) {
    gpu->memcpy_gpu_to_gpu( (size_t)devPtr, (size_t)hostPtr, size);
  } else {
    printf("GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
}

void
Simulator::cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset)
{
  g_stream_manager->push( stream_operation(src, symbol, count, offset, 0) );
}

void
Simulator::cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset)
{
  printf("GPGPU-Sim PTX: cudaMemcpyFromSymbol: symbol = %p\n", symbol);
  g_stream_manager->push( stream_operation(symbol, dst, count, offset, 0) );
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
  /*
	struct CUstream_st *s = (struct CUstream_st *)stream;
	switch( kind ) {
	case cudaMemcpyHostToDevice: g_stream_manager->push( stream_operation(src,(size_t)dst,count,s) ); break;
	case cudaMemcpyDeviceToHost: g_stream_manager->push( stream_operation((size_t)src,dst,count,s) ); break;
	case cudaMemcpyDeviceToDevice: g_stream_manager->push( stream_operation((size_t)src,(size_t)dst,count,s) ); break;
	default:
		abort();
	}
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
}


/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::cudaMemset(void *mem, int c, size_t count)
{
  CUctx_st *context = GPGPUSim_Context();
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  gpu->gpu_memset((size_t)mem, c, count);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

int
Simulator::cudaGetDeviceCount()
{
  return GPGPUSim_Init()->num_devices();
}

const struct cudaDeviceProp*
Simulator::cudaGetDeviceProperties(int device)
{
  _cuda_device_id *dev = GPGPUSim_Init();
  if (device <= dev->num_devices() ) {
    return dev->get_prop();
  } else {
    return NULL;
  }
}

void
Simulator::cudaChooseDevice(const struct cudaDeviceProp *prop)
{
  cuda_not_implemented(__my_func__,__LINE__);
}

void
Simulator::cudaSetDevice(int device)
{
  assert(device <= GPGPUSim_Init()->num_devices());
  active_device = device;
}

int
Simulator::cudaGetDevice()
{
  return active_device;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::cudaBindTexture(const struct textureReference* texref, const void *devPtr, const struct cudaChannelFormatDesc& desc, size_t size)
{
  CUctx_st *context = GPGPUSim_Context();
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in cudaBindTexture: sizeof(struct textureReference) = %zu\n", sizeof(struct textureReference));

  struct cudaArray *array;
  array = (struct cudaArray*) malloc(sizeof(struct cudaArray));
  array->desc = desc;
  array->size = size;
  array->width = size;
  array->height = 1;
  array->dimensions = 1;
  array->devPtr = (void*)devPtr;
  array->devPtr32 = (int)(long long)devPtr;

  printf("GPGPU-Sim PTX:   size = %zu\n", size);
  printf("GPGPU-Sim PTX:   texref = %p, array = %p\n", texref, array);
  printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n", gpu->gpgpu_ptx_sim_findNamefromTexture(texref));
  printf("GPGPU-Sim PTX:   ChannelFormatDesc: x=%d, y=%d, z=%d, w=%d\n", desc.x, desc.y, desc.z, desc.w);
  printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
  gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
  devPtr = (void*)(long long)array->devPtr32;
  printf("GPGPU-Sim PTX: devPtr = %p\n", devPtr);
}

void
Simulator::cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array)
{
  CUctx_st *context = GPGPUSim_Context();
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in cudaBindTextureToArray: %p %p\n", texref, array);
  printf("GPGPU-Sim PTX:   devPtr32 = %x\n", array->devPtr32);
  printf("GPGPU-Sim PTX:   Name corresponding to textureReference: %s\n", gpu->gpgpu_ptx_sim_findNamefromTexture(texref));
  printf("GPGPU-Sim PTX:   Texture Normalized? = %d\n", texref->normalized);
  gpu->gpgpu_ptx_sim_bindTextureToArray(texref, array);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::cudaGetLastError()
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif

  struct CUstream_st *s = (struct CUstream_st *)stream;
  cuda_launch_stack.push_back( kernel_config(gridDim,blockDim,sharedMem,s) );
}

void
Simulator::cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif

  gpgpusim_ptx_assert( !cuda_launch_stack.empty(), "empty launch stack" );
  kernel_config &config = cuda_launch_stack.back();
  config.set_arg(arg,size,offset);

#ifdef DEBUG_MULTIKERNEL_SIM
  /*
  fprintf(debugFile, "arg = 0x%p, size = %zu, offset = %zu\n", arg, size, offset);
  for (size_t i = 0; i < size; ++i) {
    fprintf(debugFile, "%d\n", ((const char*)arg)[offset + i]);
  }
  fprintf(debugFile, "\n");
  */
#endif
}


extern bool KAIN_Re_partition;
extern bool KAIN_profiling_phase1;
extern bool KAIN_profiling_phase2;
extern bool KAIN_profiling_phase3;
extern std::map<std::string, KAIN_IPC> KAIN_stream1;
extern std::vector<float> KAIN_stream1_ipc;
extern std::map<std::string, unsigned> KAIN_CTA_number_stream1_record;
extern std::string KAIN_kernel1;
extern bool KAIN_stream1_kernel_new_launch;
extern bool KAIN_stream1_kernel_profiling_wait_result;

extern std::map<std::string, KAIN_IPC> KAIN_stream2;
extern std::vector<float> KAIN_stream2_ipc;
extern std::map<std::string, unsigned> KAIN_CTA_number_stream2_record;
extern std::string KAIN_kernel2;
extern bool KAIN_stream2_kernel_new_launch;
extern bool KAIN_stream2_kernel_profiling_wait_result;


extern std::map<std::string, KAIN_IPC> KAIN_stream3;
extern std::map<std::string, unsigned> KAIN_CTA_number_stream3_record;
extern std::vector<float> KAIN_stream3_ipc;
extern std::string KAIN_kernel3;
extern bool KAIN_stream3_kernel_profiling;
extern bool KAIN_stream3_kernel_new_launch;
extern bool KAIN_stream3_kernel_profiling_wait_result;

extern std::map<std::string, KAIN_IPC> KAIN_stream4;
extern std::vector<float> KAIN_stream4_ipc;
extern std::map<std::string, unsigned> KAIN_CTA_number_stream4_record;
extern std::string KAIN_kernel4;
extern bool KAIN_stream4_kernel_profiling;
extern bool KAIN_stream4_kernel_new_launch;
extern bool KAIN_stream4_kernel_profiling_wait_result;


extern unsigned CTA_finished_number_stream1;
extern unsigned CTA_finished_number_stream2;
extern unsigned CTA_finished_number_stream3;
extern unsigned CTA_finished_number_stream4;


extern unsigned long long KAIN_stable_cycles;

kernel_info_t*
Simulator::cudaLaunch( const char *hostFun, ChildProcess* parent )
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif

  CUctx_st* context = GPGPUSim_Context();
  char *mode = getenv("PTX_SIM_MODE_FUNC");
  if( mode )
    sscanf(mode,"%u", &g_ptx_sim_mode);
  gpgpusim_ptx_assert( !cuda_launch_stack.empty(), "empty launch stack" );
  kernel_config config = cuda_launch_stack.back();
  struct CUstream_st *stream = config.get_stream();
  printf("\nGPGPU-Sim PTX: cudaLaunch for 0x%p (mode=%s) on stream %u\n", hostFun,
      g_ptx_sim_mode?"functional simulation":"performance simulation", stream?stream->get_uid():0 );
  kernel_info_t *grid = gpgpu_cuda_ptx_sim_init_grid(hostFun,config.get_args(),config.grid_dim(),config.block_dim(),context);
  grid->set_parent_process(parent);
  grid->check_for_prev_stats();
  std::string kname = grid->name();
  dim3 gridDim = config.grid_dim();
  dim3 blockDim = config.block_dim();
  printf("GPGPU-Sim PTX: pushing kernel \'%s\' to stream %u, gridDim= (%u,%u,%u) blockDim = (%u,%u,%u) \n",
      kname.c_str(), stream?stream->get_uid():0, gridDim.x,gridDim.y,gridDim.z,blockDim.x,blockDim.y,blockDim.z );

//////////////////KAIN profiling

/*
	
  KAIN_profiling_phase1 = true;
  KAIN_profiling_phase2 = false;
  KAIN_profiling_phase3 = false;
  KAIN_Re_partition = true;
  KAIN_stable_cycles = 0;

  if(stream->get_uid() == 1)
  {
  		KAIN_kernel1 = kname;
        CTA_finished_number_stream1 = 0;
        KAIN_stream1_kernel_new_launch = true;
  }
  else if(stream->get_uid() == 2)
  {
  		KAIN_kernel2 = kname;
	    CTA_finished_number_stream2 = 0;
        KAIN_stream2_kernel_new_launch = true;
  }
  else if(stream->get_uid() == 3)
  {
  		KAIN_kernel3 = kname;
	    CTA_finished_number_stream3 = 0;
        KAIN_stream3_kernel_new_launch = true;
  }
  else if(stream->get_uid() == 4)
  {
  		KAIN_kernel4 = kname;
	    CTA_finished_number_stream4 = 0;
        KAIN_stream4_kernel_new_launch = true;
  }
  else
        assert(0);

  //if(stream->get_uid() == 1)
  {

		KAIN_stream1.clear();
        KAIN_stream1_ipc.clear();

		{
//			KAIN_stream1_kernel_exist = false;	
//			KAIN_stream1_kernel_profiling = true;
			KAIN_stream1_kernel_profiling_wait_result = true;
		}
  
  }
//  else if (stream->get_uid() == 2)
  {
		KAIN_stream2.clear();
        KAIN_stream2_ipc.clear();
		{
			KAIN_stream2_kernel_profiling_wait_result = true;
  		}
  }

  {
		KAIN_stream3.clear();
        KAIN_stream3_ipc.clear();
		{
			KAIN_stream3_kernel_profiling_wait_result = true;
  		}
  }


  {
		KAIN_stream4.clear();
        KAIN_stream4_ipc.clear();
		{
			KAIN_stream4_kernel_profiling_wait_result = true;
  		}
  }
*/





/*
  else if (stream->get_uid() == 3)
  {
  		KAIN_kernel3 = kname;

		KAIN_stream3.erase(kname);
		KAIN_CTA_number_stream3_record.erase(kname);

	 	if(KAIN_stream3.find(kname) != KAIN_stream3.end())//found
		{
//			KAIN_stream2_kernel_exist = true;
			printf("KAIN Kernel: %s, found, no need profiling\n",kname.c_str());
			KAIN_stream3_kernel_profiling = false;
			KAIN_stream3_kernel_profiling_wait_result = false;
			CTA_finished_number_stream3 = KAIN_CTA_number_stream3_record.find(kname)->second;
		}
		else
		{
//			KAIN_stream2_kernel_exist = false;	
			printf("KAIN Kernel: %s, could not find, profiling\n",kname.c_str());
			KAIN_stream3_kernel_profiling = true;
			CTA_finished_number_stream3 = 0;
			KAIN_stream3_kernel_profiling_wait_result = true;

  		}
  }
  else if (stream->get_uid() == 4)
  {
  		KAIN_kernel4 = kname;

		KAIN_stream4.erase(kname);
		KAIN_CTA_number_stream4_record.erase(kname);

	 	if(KAIN_stream4.find(kname) != KAIN_stream4.end())//found
		{
//			KAIN_stream2_kernel_exist = true;
			printf("KAIN Kernel: %s, found, no need profiling\n",kname.c_str());
			KAIN_stream4_kernel_profiling = false;
			KAIN_stream4_kernel_profiling_wait_result = false;
			CTA_finished_number_stream4 = KAIN_CTA_number_stream4_record.find(kname)->second;
		}
		else
		{
//			KAIN_stream2_kernel_exist = false;	
			printf("KAIN Kernel: %s, could not find, profiling\n",kname.c_str());
			KAIN_stream4_kernel_profiling = true;
			CTA_finished_number_stream4 = 0;
			KAIN_stream4_kernel_profiling_wait_result = true;

  		}
  }
  else
  	assert(0);
*/
  
//////////////////////

  stream_operation op(grid,g_ptx_sim_mode,stream);
  g_stream_manager->push(op);
  cuda_launch_stack.pop_back();

  return grid;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::cudaThreadSynchronize(void)
{
  synchronize();
};

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::__cudaRegisterFatBinary(unsigned long long fat_cubin_handle, const char* ptx)
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif

  static unsigned source_num=1;

  CUctx_st *context = GPGPUSim_Context();
  if(context->get_device()->get_gpgpu()->get_config().convert_to_ptxplus() ) {
    printf("GPGPU-Sim PTX: ERROR ** PTXPlus is only supported through cuobjdump\n"
        "\tEither enable cuobjdump or disable PTXPlus in your configuration file\n");
    exit(1);
  }

  symbol_table* symtab=gpgpu_ptx_sim_load_ptx_from_string(ptx,source_num);
  context->add_binary(symtab,fat_cubin_handle);
  gpgpu_ptxinfo_load_from_string( ptx, source_num );
  source_num++;
  load_static_globals(symtab,STATIC_ALLOC_LIMIT,0xFFFFFFFF,context->get_device()->get_gpgpu());
  load_constants(symtab,STATIC_ALLOC_LIMIT,context->get_device()->get_gpgpu());
}

void
Simulator::__cudaRegisterFunction(unsigned fat_cubin_handle,
                                  const char* hostFun, char* deviceFun)
{
#ifdef DEBUG_MULTIKERNEL_SIM
  PRINT_CALL;
#endif

  CUctx_st *context = GPGPUSim_Context();
  printf("GPGPU-Sim PTX: __cudaRegisterFunction %s : hostFun 0x%p, fat_cubin_handle = %u\n",
      deviceFun, hostFun, fat_cubin_handle);
//  assert(!context->get_device()->get_gpgpu()->get_config().use_cuobjdump());
  context->register_function( fat_cubin_handle, hostFun, deviceFun );
}

void
Simulator::__cudaRegisterTexture(const struct textureReference *hostVar, const char* deviceName, int dim, int norm, int ext)
{
  CUctx_st *context = GPGPUSim_Context();
  gpgpu_t *gpu = context->get_device()->get_gpgpu();
  printf("GPGPU-Sim PTX: in __cudaRegisterTexture:\n");
  gpu->gpgpu_ptx_sim_bindNameToTexture(deviceName, hostVar, dim, norm, ext);
  printf("GPGPU-Sim PTX:   int dim = %d\n", dim);
  printf("GPGPU-Sim PTX:   int norm = %d\n", norm);
  printf("GPGPU-Sim PTX:   int ext = %d\n", ext);
  printf("GPGPU-Sim PTX:   Execution warning: Not finished implementing \"%s\"\n", __my_func__ );
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

// Multikernel Simulator uses streams to differentiate kernels between
// different applications.
// Therefore, it is currently forbidden for applications to use cudaStream
cudaStream_t
Simulator::cudaStreamCreate()
{
  cudaStream_t stream = new struct CUstream_st();
  g_stream_manager->add_stream(stream);
  return stream;
}

void
Simulator::cudaStreamDestroy(cudaStream_t stream)
{
  g_stream_manager->destroy_stream(stream);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

void
Simulator::cudaFuncSetCacheConfig(const char * hostFunc, enum cudaFuncCache cacheConfig)
{
  CUctx_st *context = GPGPUSim_Context();
  context->get_device()->get_gpgpu()->set_cache_config(context->get_kernel(hostFunc)->get_name(), (FuncCache)cacheConfig);
}

/****************************************/
/* Simulate method                      */
/****************************************/
void
Simulator::initialize_streams(const unsigned numProcesses)
{
  // make sure we initialize GPGPU-sim
  GPGPUSim_Context();

  assert(mapped_streams.empty());
  // all the kernels go to non-zero streams (meaning not default stream)
  for (unsigned i = 0; i < numProcesses; ++i) {
    printf("GPGPU-Sim Stream: Creating stream\n");
    mapped_streams.push_back(cudaStreamCreate());
  }
}

void
Simulator::initialize_scheduler(MKScheduler* mk_sched)
{
  CUctx_st *context = GPGPUSim_Context();
  gpgpu_sim *gpu = context->get_device()->get_gpgpu();

  mk_sched->set_number_of_SMs(gpu->get_config().num_shader());
  gpu->set_mk_scheduler(mk_sched);
}

void
Simulator::get_ready_for_launch_or_terminate(ChildProcess* process)
{
  bool is_kernel_launched = false;
  while (!is_kernel_launched) {
    is_kernel_launched = process_message(process);
  }
}

void
Simulator::launch_bogus_kernel(ChildProcess* process)
{
  ChildProcess::kernel_build_info* bogus_kernel = process->get_next_bogus_kernel_and_rotate();
  process->set_launched_kernel();
  launch_kernels.push_back(std::make_pair(process, bogus_kernel->hostFun));
  cuda_launch_stack.push_back( bogus_kernel->config );
}

bool
Simulator::launch(LauncherOptionParser* opt)
{
  while (!launch_kernels.empty()) {
    // cuda_launch_stack is in LIFO order
    // launch_kernels have to be handled in the same order
    std::pair<ChildProcess*, const char*> & curr_kernel = launch_kernels.back();
    kernel_info_t* kernel_info = cudaLaunch(curr_kernel.second, curr_kernel.first);
    // moved inside to cudaLaunch
    //kernel_info->set_parent_process(curr_kernel.first);
    printf("GPGPU-Sim Kernel: Process %d initiating kernel @ 0x%llx..\n", curr_kernel.first->getID(), (unsigned long long) kernel_info);
    launch_kernels.pop_back();
  }

  kernel_info_t* finished_kernel = gpgpu_sim_progress(opt);
  if (finished_kernel) {
    printf("GPGPU-Sim Kernel: Finishing kernel of process %d @ 0x%llx..\n", finished_kernel->get_parent_process()->getID(), (unsigned long long) finished_kernel);
    finished_kernel->get_parent_process()->unset_launched_kernel();
    opt->getScheduler()->remove_kernel(finished_kernel);
    delete finished_kernel;

    return true;
  }

  return false;
}

// returns true if the message was cudaLaunch or exit
// otherwise, returns false
bool
Simulator::process_message(ChildProcess* process)
{
  Communicate* pipe = process->get_pipe();
  printf("KAIN: pid %d, get message and process it\n",getpid());
  fflush(stdout);
  switch (process->get_message_header()) {
    case MESSAGE_GPU_MALLOC:
    {
      size_t size;
      pipe->read(size);
      pipe->write(cudaMalloc(size));
      break;
    }

    case MESSAGE_GPU_MALLOCARRAY:
    {
      size_t size;
      pipe->read(size);
      pipe->write(cudaMallocArray(size));
      break;
    }

    case MESSAGE_GPU_MEMCPY:
    {
      enum cudaMemcpyKind kind;
      size_t count;
      pipe->read(kind);
      pipe->read(count);

      if ( kind == cudaMemcpyHostToDevice ) {
        void* dst = NULL;
        void* src = malloc(count);

        pipe->read(dst);
        pipe->read(src, count);

        cudaMemcpy(dst, src, count, kind);
        free(src);

      } else if ( kind == cudaMemcpyDeviceToHost ) {
        void* dst = malloc(count);
        void* src = NULL;

        pipe->read(src);
        cudaMemcpy(dst, src, count, kind);

        pipe->write(dst, count);
        free(dst);

      } else if ( kind == cudaMemcpyDeviceToDevice ) {
        void* dst = NULL;
        void* src = NULL;

        pipe->read(src);
        pipe->read(dst);

        cudaMemcpy(dst, src, count, kind);

      } else {
        printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
        abort();
      }
      break;
    }

    case MESSAGE_GPU_MEMCPY_SYMBOL:
    {
      enum cudaMemcpyKind kind;
      size_t count;
      size_t offset;
      pipe->read(kind);
      pipe->read(count);
      pipe->read(offset);

      char* orgSymbol = NULL;
      pipe->read(orgSymbol);
      char* symbolPtr = (char*)process->get_unique_pointer((void*)orgSymbol);

      size_t length;
      pipe->read(length);
      char* symbol = (char*)malloc(length);
      pipe->read(symbol, length);

      if (!process->hasVarName(symbolPtr)) {
        process->addVarName((const void*)symbolPtr, symbol);
      } else {
        free(symbol);
        symbol = process->findVarName(symbolPtr);
      }

      if ( kind == cudaMemcpyHostToDevice ) {
        void* src = malloc(count);
        pipe->read(src, count);

        cudaMemcpyToSymbol(symbol, src, count, offset);
        free(src);

      } else if ( kind == cudaMemcpyDeviceToHost ) {
        void* dst = malloc(count);
        cudaMemcpyFromSymbol(dst, symbol, count, offset);

        pipe->write(dst, count);
        free(dst);

      } else {
        printf("GPGPU-Sim PTX: cudaMemcpySymbol - ERROR : unsupported cudaMemcpyKind\n");
        abort();
      }
      break;
    }

    case MESSAGE_GPU_MEMCPY_2D_TO_ARRAY:
    {
      enum cudaMemcpyKind kind;
      size_t size;
      pipe->read(kind);
      pipe->read(size);

      if (kind == cudaMemcpyHostToDevice) {
        void* dst;
        void* src = malloc(size);

        // device pointer
        pipe->read(dst);
        // host src
        pipe->read(src, size);
        cudaMemcpy2DToArray(dst, src, kind, size);
      } else if (kind == cudaMemcpyDeviceToHost) {
      } else {
        assert(kind == cudaMemcpyDeviceToDevice);
        void* dst;
        void* src;

        // device ptr
        pipe->read(dst);
        // device ptr
        pipe->read(src);
        cudaMemcpy2DToArray(dst, src, kind, size);
      }
      break;
    }

    case MESSAGE_GPU_MEMSET:
    {
      void* mem = NULL;
      int c;
      size_t count;

      pipe->read(mem);
      pipe->read(c);
      pipe->read(count);

      cudaMemset(mem, c, count);
      break;
    }

    case MESSAGE_GPU_GET_DEVICE_COUNT:
    {
      pipe->write(cudaGetDeviceCount());
      break;
    }

    case MESSAGE_GPU_GET_DEVICE_PROPERTY:
    {
      int device;
      pipe->read(device);
      const struct cudaDeviceProp* prop = cudaGetDeviceProperties(device);
      const bool success = (prop != NULL);
      pipe->write(success);
      if (success) {
        pipe->write(*prop);
      }
      break;
    }

    case MESSAGE_GPU_SET_DEVICE:
    {
      int device;
      pipe->read(device);
      cudaSetDevice(device);
      break;
    }

    case MESSAGE_GPU_GET_DEVICE:
    {
      int device = cudaGetDevice();
      pipe->write(device);
      break;
    }

    case MESSAGE_GPU_BIND_TEXTURE:
    {
      // texref
      struct textureReference* orgTexRef;
      pipe->read(orgTexRef);
      struct textureReference* texRefPtr = (struct textureReference*)process->get_unique_pointer((void*)orgTexRef);

      struct textureReference* texref = process->findTextureReference(texRefPtr);
      pipe->read(*texref);

      // devicePtr
      void* devPtr;
      pipe->read(devPtr);

      // desc
      struct cudaChannelFormatDesc desc;
      pipe->read(desc);

      // size
      size_t size;
      pipe->read(size);

      cudaBindTexture(texref, devPtr, desc, size);
      break;
    }

    case MESSAGE_GPU_BIND_TEXTURE_TO_ARRAY:
    {
      // texref
      struct textureReference* orgTexRef;
      pipe->read(orgTexRef);
      struct textureReference* texRefPtr = (struct textureReference*)process->get_unique_pointer((void*)orgTexRef);

      struct textureReference* texref = process->findTextureReference(texRefPtr);
      pipe->read(*texref);

      // array
      struct cudaArray* array = (struct cudaArray*) malloc(sizeof(struct cudaArray));
      pipe->read(*array);

      cudaBindTextureToArray(texref, array);
    }

    case MESSAGE_GPU_GET_LAST_ERROR:
    {
      cudaGetLastError();
      break;
    }

    case MESSAGE_GPU_CONFIGURE_CALL:
    {
      dim3 gridDim;
      dim3 blockDim;
      size_t sharedMem;

      pipe->read(gridDim);
      pipe->read(blockDim);
      pipe->read(sharedMem);
      // CUstream_st is ignored

      // stream is unique per process
      cudaConfigureCall(gridDim, blockDim, sharedMem, mapped_streams[process->getID()]);
      //cudaConfigureCall(gridDim, blockDim, sharedMem, NULL);
      break;
    }

    case MESSAGE_GPU_SETUP_ARGUMENT:
    {
      size_t size;
      pipe->read(size);

      char* arg = (char*)malloc(size + 1);
      pipe->read(arg, size);
      arg[size] = '\0';

      cudaSetupArgument(arg, size, 0);
      // FIXME: arg has to be freed somewhere!
      break;
    }

    case MESSAGE_GPU_LAUNCH:
    {
      char* orgHostFun;
      pipe->read(orgHostFun);
      char* hostFun = (char*)process->get_unique_pointer((void*)orgHostFun);

      // no longer launches kernel directly
      //cudaLaunch(hostFun);
      process->set_launched_kernel();
      launch_kernels.push_back(std::make_pair(process, hostFun));

      // this inserts kernel to the child process for bogus launch
      process->add_kernel(hostFun, cuda_launch_stack.back());
      return true;
    }

    case MESSAGE_GPU_SYNCHRONIZE:
    {
      cudaThreadSynchronize();
      break;
    }

    case MESSAGE_GPU_REGISTER_FAT_BINARY:
    {
      unsigned forced_max_capability = GPGPUSim_Context()->get_device()->get_gpgpu()->get_config().get_forced_max_capability();
      unsigned long long fat_cubin_handle = get_fat_cubin_handle();


  	printf("KAIN: pid %d, write parameter to the benchmark\n",getpid());
  	fflush(stdout);

      pipe->write(forced_max_capability);
      pipe->write(fat_cubin_handle);

      bool found;
      pipe->read(found);

      if (found) {
        size_t length;
        pipe->read(length);

        char* ptx = (char*)malloc(length);
        pipe->read(ptx, length);

#ifdef DEBUG_MULTIKERNEL_SIM
        fprintf(debugFile, "%s\n", ptx);
#endif

        __cudaRegisterFatBinary(fat_cubin_handle, ptx);
      } else {
        printf("GPGPU-Sim PTX: warning -- did not find an appropriate PTX in cubin\n");
      }

      process->inc_fat_binary();
      break;
    }

    case MESSAGE_GPU_REGISTER_FUNCTION:
    {
      unsigned fat_cubin_handle;
      char* orgHostFun;

      pipe->read(fat_cubin_handle);
      pipe->read(orgHostFun);
      char* hostFun = (char*)process->get_unique_pointer((void*)orgHostFun);

      size_t length;
      pipe->read(length);
      char* deviceFun = (char*)malloc(length);
      pipe->read(deviceFun, length);

      __cudaRegisterFunction(fat_cubin_handle, hostFun, deviceFun);
      break;
    }

    case MESSAGE_GPU_REGISTER_CONST_VARIABLE:
    {
      size_t length;

      char* orgHostVar;
      pipe->read(orgHostVar);
      char* hostVarPtr = (char*)process->get_unique_pointer((void*)orgHostVar);
      pipe->read(length);
      char* hostVar = (char*)malloc(length);
      pipe->read(hostVar, length);

      process->addVarName((const void*)hostVarPtr, hostVar);

      pipe->read(length);
      char* deviceName = (char*)malloc(length);
      pipe->read(deviceName, length);

      int size;
      pipe->read(size);

      gpgpu_ptx_sim_register_const_variable(hostVar, deviceName, size);
      break;
    }

    case MESSAGE_GPU_REGISTER_GLOBAL_VARIABLE:
    {
      size_t length;

      char* orgHostVar;
      pipe->read(orgHostVar);
      char* hostVarPtr = (char*)process->get_unique_pointer((void*)orgHostVar);
      pipe->read(length);
      char* hostVar = (char*)malloc(length);
      pipe->read(hostVar, length);

      process->addVarName((const void*)hostVarPtr, hostVar);

      pipe->read(length);
      char* deviceName = (char*)malloc(length);
      pipe->read(deviceName, length);

      int size;
      pipe->read(size);

      gpgpu_ptx_sim_register_global_variable(hostVar, deviceName, size);
      break;
    }

    case MESSAGE_GPU_REGISTER_TEXTURE:
    {
      // hostVar
      struct textureReference* orgHostVar;
      pipe->read(orgHostVar);
      struct textureReference* hostVarPtr = (struct textureReference*)process->get_unique_pointer((void*)orgHostVar);
      struct textureReference* hostVar = new textureReference();
      pipe->read(*hostVar);
      process->addTextureReference((const void*)hostVarPtr, hostVar);

      // deviceName
      size_t length;
      pipe->read(length);
      char* deviceName = (char*)malloc(length);
      pipe->read(deviceName, length);

      int dim, norm, ext;
      pipe->read(dim);
      pipe->read(norm);
      pipe->read(ext);

      __cudaRegisterTexture(hostVar, deviceName, dim, norm, ext);
      // we do not need deviceName
      // it is converted to std::string inside cuda-sim
      free(deviceName);
      break;
    }

    case MESSAGE_GPU_SET_CACHE_CONFIG:
    {
      char* orgHostFun;
      enum cudaFuncCache cacheConfig;

      // pointer itself is important
      pipe->read(orgHostFun);
      pipe->read(cacheConfig);

      char* hostFun = (char*)process->get_unique_pointer((void*)orgHostFun);
      cudaFuncSetCacheConfig(hostFun, cacheConfig);
      break;
    }

    case MESSAGE_GPU_EXIT_SIMULATION:
    {
      process->dec_fat_binary();
      const bool noMoreBinary = process->no_more_fat_binary();
      pipe->write(noMoreBinary);
      if (noMoreBinary) {
        printf("GPGPU-Sim MK-Sim: Terminating process %d\n", process->getID());
        process->terminate();
        //process->rewind();
        //exit_simulation();
        return true;
      }

      printf("GPGPU-Sim MK-Sim: Remaining %d fat binaries for process %d\n", process->get_num_fat_binary(), process->getID());
      break;
    }

    case MESSAGE_GPU_CURR_CYCLE:
    {
      pipe->write(get_curr_cycle());
      break;
    }

    default:
      printf("GPGPU-Sim Message: Unknown message.\n");
      abort();
  }

  return false;
}

unsigned long long
Simulator::get_fat_cubin_handle()
{
  static unsigned next_fat_bin_handle = 1;
  return next_fat_bin_handle++;
}


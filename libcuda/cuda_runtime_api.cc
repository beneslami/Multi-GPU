// This file created from cuda_runtime_api.h distributed with CUDA 1.1
// Changes Copyright 2009,  Tor M. Aamodt, Ali Bakhoda and George L. Yuan
// University of British Columbia

/* 
 * cuda_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan and the University of British Columbia, Vancouver, 
 * BC V6T 1Z4, All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <stdarg.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <map>
#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h> // Apple's version of GLUT is here
#else
#include <GL/gl.h>
#endif
#endif

#define __CUDA_RUNTIME_API_H__

#include "host_defines.h"
#include "builtin_types.h"
#include "driver_types.h"
#include "__cudaFatFormat.h"
#include "../launcher/communicate.h"
//#include "../src/gpgpu-sim/gpu-sim.h"
//#include "../src/cuda-sim/ptx_loader.h"
//#include "../src/cuda-sim/cuda-sim.h"
//#include "../src/cuda-sim/ptx_ir.h"
//#include "../src/cuda-sim/ptx_parser.h"
//#include "../src/gpgpusim_entrypoint.h"
#include "../src/stream_manager.h"

#include <pthread.h>
#include <semaphore.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// jasonjk begin
// define this to see the trace of function calls
#define RECORD_FUNCTION_CALLS

#ifdef RECORD_FUNCTION_CALLS
#define PRINT_CALL printf("%s\n", __my_func__); \
                    fflush(stdout)
#else
#define PRINT_CALL
#endif
// jasonjk end

/*DEVICE_BUILTIN*/
struct cudaArray
{
	void *devPtr;
	int devPtr32;
	struct cudaChannelFormatDesc desc;
	int width;
	int height;
	int size; //in bytes
	unsigned dimensions;
};

#if !defined(__dv)
#if defined(__cplusplus)
#define __dv(v) \
		= v
#else /* __cplusplus */
#define __dv(v)
#endif /* __cplusplus */
#endif /* !__dv */

cudaError_t g_last_cudaError = cudaSuccess;

// jasonjk begin
// no longer used
//extern stream_manager *g_stream_manager;
// jasonjk end

#if defined __APPLE__
#   define __my_func__    __PRETTY_FUNCTION__
#else
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __my_func__    __PRETTY_FUNCTION__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __my_func__    __func__
#  else
#   define __my_func__    ((__const char *) 0)
#  endif
# endif
#endif

// jasonjk begin
static Communicate *g_pipe = NULL;

void Pipe_Init()
{
  if (g_pipe == NULL) {
    g_pipe = new Communicate();
  }
}

void Pipe_Finish()
{
  if (g_pipe != NULL) {
    delete g_pipe;
    g_pipe = NULL;
  }
}
// jasonjk end

// these functions are no longer needed
// they are processed in the main process GPGPU-sim
//class _cuda_device_id *GPGPUSim_Init()
//static CUctx_st* GPGPUSim_Context()

static void cuda_not_implemented( const char* func, unsigned line )
{
  fflush(stdout);
  fflush(stderr);
  printf("\n\nGPGPU-Sim PTX: Execution error: CUDA API function \"%s()\" has not been implemented yet.\n"
         "                 [$GPGPUSIM_ROOT/libcuda/%s around line %u]\n\n\n",
         func,__FILE__, line );
  fflush(stdout);
  abort();
}

typedef std::map<unsigned,CUevent_st*> event_tracker_t;
event_tracker_t g_timer_events;

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

extern "C" {

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_MALLOC);
  g_pipe->write(size);
  g_pipe->read(*devPtr);

  if ( *devPtr  ) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size)
{
  PRINT_CALL;
  *ptr = malloc(size);
  if ( *ptr  ) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}
__host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
  PRINT_CALL;
  unsigned malloc_width_inbytes = width;
  g_pipe->write(MESSAGE_GPU_MALLOC);
  g_pipe->write(malloc_width_inbytes*height);
  g_pipe->read(*devPtr);
  pitch[0] = malloc_width_inbytes;
  if ( *devPtr  ) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(1))
{
  PRINT_CALL;
  size_t size = width * height * ((desc->x + desc->y + desc->z + desc->w)/8);

  (*array) = (struct cudaArray*) malloc(sizeof(struct cudaArray));
  (*array)->desc = *desc;
  (*array)->width = width;
  (*array)->height = height;
  (*array)->size = size;
  (*array)->dimensions = 2;

  g_pipe->write(MESSAGE_GPU_MALLOCARRAY);
  g_pipe->write(size);
  g_pipe->read((*array)->devPtr);

  ((*array)->devPtr32)= (int) (long long)((*array)->devPtr);
  printf("GPGPU-Sim PTX: cudaMallocArray: devPtr32 = %d\n", ((*array)->devPtr32));
  if ( ((*array)->devPtr) ) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorMemoryAllocation;
  }
}

__host__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
  PRINT_CALL;
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
  PRINT_CALL;
  free (ptr);  // this will crash the system if called twice
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array)
{
  PRINT_CALL;
  // TODO...  manage g_global_mem space?
  return g_last_cudaError = cudaSuccess;
};


/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_MEMCPY);
  g_pipe->write(kind);
  g_pipe->write(count);

  if ( kind == cudaMemcpyHostToDevice ) {
    g_pipe->write(dst);
    g_pipe->write(src, count);
  } else if ( kind == cudaMemcpyDeviceToHost ) {
    g_pipe->write(src);
    g_pipe->read(dst, count);
  } else if ( kind == cudaMemcpyDeviceToDevice ) {
    g_pipe->write(src);
    g_pipe->write(dst);
  } else {
    printf("GPGPU-Sim PTX: cudaMemcpy - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_MEMCPY);
  g_pipe->write(kind);
  g_pipe->write(count);

  if ( kind == cudaMemcpyHostToDevice ) {
    g_pipe->write(dst->devPtr);
    g_pipe->write(src, count);
  } else if ( kind == cudaMemcpyDeviceToHost ) {
    g_pipe->write(src);
    g_pipe->read(dst->devPtr, count);
  } else if ( kind == cudaMemcpyDeviceToDevice ) {
    g_pipe->write(src);
    g_pipe->write(dst->devPtr);
  } else {
    printf("GPGPU-Sim PTX: cudaMemcpyToArray - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }
  return g_last_cudaError = cudaSuccess;
}


__host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
  PRINT_CALL;
  /*
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
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
  PRINT_CALL;
  size_t size = spitch * height;
  size_t channel_size = dst->desc.w + dst->desc.x + dst->desc.y + dst->desc.z;
  assert( ((channel_size % 8) == 0) && "none byte multiple destination channel size not supported");
  unsigned elem_size = channel_size / 8;
  assert( (dst->dimensions == 2) && "copy to none 2D array not supported" );
  assert( (wOffset == 0) && "non-zero wOffset not yet supported" );
  assert( (hOffset == 0) && "non-zero hOffset not yet supported" );
  assert( (dst->height == (int)height) && "partial copy not supported" );
  assert( (elem_size * dst->width == width) && "partial copy not supported" );
  assert( (spitch == width) && "spitch != width not supported" );

  g_pipe->write(MESSAGE_GPU_MEMCPY_2D_TO_ARRAY);
  g_pipe->write(kind);
  g_pipe->write(size);

  if( kind == cudaMemcpyHostToDevice ) {
    g_pipe->write(dst->devPtr);
    g_pipe->write(src, size);
  } else if( kind == cudaMemcpyDeviceToHost ) {
    //g_pipe->write(dst->devPtr);
    //g_pipe->read(src, size);
    //gpu->memcpy_from_gpu( dst->devPtr, (size_t)src, size);
    cuda_not_implemented(__my_func__,__LINE__);
    return g_last_cudaError = cudaErrorUnknown;
  } else if( kind == cudaMemcpyDeviceToDevice ) {
    g_pipe->write(dst->devPtr);
    g_pipe->write(src);
  } else {
    printf("GPGPU-Sim PTX: cudaMemcpy2D - ERROR : unsupported cudaMemcpyKind\n");
    abort();
  }

  dst->devPtr32 = (unsigned) (size_t)(dst->devPtr);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice))
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice))
{
  PRINT_CALL;
  assert(kind == cudaMemcpyHostToDevice);
  printf("GPGPU-Sim PTX: cudaMemcpyToSymbol: symbol = %p, name = %s\n", symbol, symbol);
  fflush(stdout);

  g_pipe->write(MESSAGE_GPU_MEMCPY_SYMBOL);
  g_pipe->write(kind);
  g_pipe->write(count);
  g_pipe->write(offset);

  // symbol pointer itself, and contents are important
  g_pipe->write(symbol);
  size_t length= strlen(symbol) + 1;
  g_pipe->write(length);
  g_pipe->write(symbol, length);
  // copy src
  g_pipe->write(src, count);
  return g_last_cudaError = cudaSuccess;
}


__host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost))
{
  PRINT_CALL;
  assert(kind == cudaMemcpyDeviceToHost);

  g_pipe->write(MESSAGE_GPU_MEMCPY_SYMBOL);
  g_pipe->write(kind);
  g_pipe->write(count);
  g_pipe->write(offset);

  // symbol pointer itself, and contents are important
  g_pipe->write(symbol);
  size_t length= strlen(symbol) + 1;
  g_pipe->write(length);
  g_pipe->write(symbol, length);
  // get data
  g_pipe->read(dst, count);
  return g_last_cudaError = cudaSuccess;
}


/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
  PRINT_CALL;
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
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_MEMSET);
  g_pipe->write(mem);
  g_pipe->write(c);
  g_pipe->write(count);

  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c, size_t width, size_t height)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}



/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const char *symbol)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


__host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const char *symbol)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}


/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_GET_DEVICE_COUNT);
  g_pipe->read(*count);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
  PRINT_CALL;
  bool success;
  g_pipe->write(MESSAGE_GPU_GET_DEVICE_PROPERTY);
  g_pipe->write(device);
  g_pipe->read(success);
  if (success) {
    g_pipe->read(*prop);
  } else {
    return g_last_cudaError = cudaErrorInvalidDevice;
  }
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaSetDevice(int device)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_SET_DEVICE);
  g_pipe->write(device);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_GET_DEVICE);
  g_pipe->read(*device);
  return g_last_cudaError = cudaSuccess;
}


/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset,
		const struct textureReference *texref,
		const void *devPtr,
		const struct cudaChannelFormatDesc *desc,
		size_t size __dv(UINT_MAX))
{
  PRINT_CALL;

  g_pipe->write(MESSAGE_GPU_BIND_TEXTURE);
  // texref, pointer and contents are both important
  g_pipe->write(texref);
  g_pipe->write(*texref);
  // devPtr, pointer is important
  g_pipe->write(devPtr);
  // desc, content is important
  g_pipe->write(*desc);
  // size, content is important
  g_pipe->write(size);
  return g_last_cudaError = cudaSuccess;
}


__host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
  PRINT_CALL;
  printf("Child: GPGPU-Sim PTX: in cudaBindTextureToArray: %p %p\n", texref, array);

  g_pipe->write(MESSAGE_GPU_BIND_TEXTURE_TO_ARRAY);
  // texref, pointer and contents are both important
  g_pipe->write(texref);
  g_pipe->write(*texref);
  // array contents is important
  g_pipe->write(*array);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref)
{
  PRINT_CALL;
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const char *symbol)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)
{
  PRINT_CALL;
  *desc = array->desc;
  return g_last_cudaError = cudaSuccess;
}

__host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
  PRINT_CALL;

  struct cudaChannelFormatDesc dummy;
  dummy.x = x;
  dummy.y = y;
  dummy.z = z;
  dummy.w = w;
  dummy.f = f;
  return dummy;
}

__host__ cudaError_t CUDARTAPI cudaGetLastError(void)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_GET_LAST_ERROR);
  return g_last_cudaError;
}

__host__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error)
{
  PRINT_CALL;

  if( g_last_cudaError == cudaSuccess )
    return "no error";
  char buf[1024];
  snprintf(buf,1024,"<<GPGPU-Sim PTX: there was an error (code = %d)>>", g_last_cudaError);
  return strdup(buf);
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
// These functions invoke a kernel

__host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_CONFIGURE_CALL);
  g_pipe->write(gridDim);
  g_pipe->write(blockDim);
  g_pipe->write(sharedMem);
  // CUstream_st has std::list<stream_operation>, and pthread_mutex_t m_lock
  //struct CUstream_st *s = (struct CUstream_st *)stream;
  //g_pipe->write(*s);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_SETUP_ARGUMENT);
  g_pipe->write(size);
  g_pipe->write(arg, size);
  // offset is not meaningful
  return g_last_cudaError = cudaSuccess;
}


__host__ cudaError_t CUDARTAPI cudaLaunch( const char *hostFun )
{
  PRINT_CALL;
  g_pipe->write(MESSAGE_GPU_LAUNCH);
  g_pipe->write(hostFun);
  return g_last_cudaError = cudaSuccess;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream)
{
  PRINT_CALL;
  /*
	printf("GPGPU-Sim PTX: cudaStreamCreate\n");
#if (CUDART_VERSION >= 3000)
	*stream = new struct CUstream_st();
	g_stream_manager->add_stream(*stream);
#else
	*stream = 0;
	printf("GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported (%s)\n", __my_func__);
#endif
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
  PRINT_CALL;
  /*
#if (CUDART_VERSION >= 3000)
	g_stream_manager->destroy_stream(stream);
#endif
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
{
  PRINT_CALL;
  /*
#if (CUDART_VERSION >= 3000)
	if( stream == NULL )
		return g_last_cudaError = cudaErrorInvalidResourceHandle;
	stream->synchronize();
#else
	printf("GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported (%s)\n", __my_func__);
#endif
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

__host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream)
{
  PRINT_CALL;
  /*
#if (CUDART_VERSION >= 3000)
	if( stream == NULL )
		return g_last_cudaError = cudaErrorInvalidResourceHandle;
	return g_last_cudaError = stream->empty()?cudaSuccess:cudaErrorNotReady;
#else
	printf("GPGPU-Sim PTX: WARNING: Asynchronous kernel execution not supported (%s)\n", __my_func__);
	return g_last_cudaError = cudaSuccess; // it is always success because all cuda calls are synchronous
#endif
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event)
{
  PRINT_CALL;
  CUevent_st *e = new CUevent_st(false);
  g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
  *event = e;
#else
  *event = e->get_uid();
#endif
  return g_last_cudaError = cudaSuccess;
}

CUevent_st *get_event(cudaEvent_t event)
{
  unsigned event_uid;
#if CUDART_VERSION >= 3000
  event_uid = event->get_uid();
#else
  event_uid = event;
#endif
  event_tracker_t::iterator e = g_timer_events.find(event_uid);
  if( e == g_timer_events.end() )
    return NULL;
  return e->second;
}

__host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  PRINT_CALL;
  CUevent_st *e = get_event(event);
  if ( !e ) return g_last_cudaError = cudaErrorUnknown;

  unsigned long long curr_cycle;
  g_pipe->write(MESSAGE_GPU_CURR_CYCLE);
  g_pipe->read(curr_cycle);

  time_t wallclock = time((time_t *)NULL);
  e->update( curr_cycle, wallclock);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event)
{
  PRINT_CALL;
  CUevent_st *e = get_event(event);
  if ( e == NULL ) {
    return g_last_cudaError = cudaErrorInvalidValue;
  } else if ( e->done() ) {
    return g_last_cudaError = cudaSuccess;
  } else {
    return g_last_cudaError = cudaErrorNotReady;
  }
}

__host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event)
{
  PRINT_CALL;
  printf("GPGPU-Sim API: cudaEventSynchronize ** waiting for event\n");
  fflush(stdout);
  CUevent_st *e = (CUevent_st*) event;
  assert(e->done());
  printf("GPGPU-Sim API: cudaEventSynchronize ** event detected\n");
  fflush(stdout);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
{
  PRINT_CALL;
  CUevent_st *e = get_event(event);
  unsigned event_uid = e->get_uid();
  event_tracker_t::iterator pe = g_timer_events.find(event_uid);
  if ( pe == g_timer_events.end() )
    return g_last_cudaError = cudaErrorInvalidValue;
  g_timer_events.erase(pe);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
  PRINT_CALL;
  time_t elapsed_time;
  CUevent_st *s = get_event(start);
  CUevent_st *e = get_event(end);
  if ( s == NULL || e == NULL )
    return g_last_cudaError = cudaErrorUnknown;
  elapsed_time = e->clock() - s->clock();
  *ms = 1000*elapsed_time;
  return g_last_cudaError = cudaSuccess;
}



/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

__host__ cudaError_t CUDARTAPI cudaThreadExit(void)
{
  PRINT_CALL;

  //g_pipe->write(MESSAGE_GPU_EXIT_SIMULATION);
  return g_last_cudaError = cudaSuccess;
}

__host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void)
{
  PRINT_CALL;
  //Called on host side
  g_pipe->write(MESSAGE_GPU_SYNCHRONIZE);
  return g_last_cudaError = cudaSuccess;
};

int CUDARTAPI __cudaSynchronizeThreads(void**, void*)
{
  PRINT_CALL;
  return cudaThreadExit();
}



/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

#if (CUDART_VERSION >= 3010)

typedef struct CUuuid_st {                                /**< CUDA definition of UUID */
    char bytes[16];
} CUuuid;

/**
 * CUDA UUID types
 */
// typedef __device_builtin__ struct CUuuid_st cudaUUID_t;

__host__ cudaError_t CUDARTAPI cudaGetExportTable(const void **ppExportTable, const cudaUUID_t *pExportTableId)
{
  PRINT_CALL;
  /*
	printf("cudaGetExportTable: UUID = "); 
	for (int s = 0; s < 16; s++) {
		printf("%#2x ", (unsigned char) (pExportTableId->bytes[s])); 
	}
	printf("\n"); 
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

#endif

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/

 
enum cuobjdumpSectionType {
    PTXSECTION=0,
    ELFSECTION
};


class cuobjdumpSection {
public:
    //Constructor
    cuobjdumpSection() {
        arch = 0; 
        identifier = "";
    }    
    virtual ~cuobjdumpSection() {}
    unsigned getArch() {return arch;}
    void setArch(unsigned a) {arch = a;}
    std::string getIdentifier() {return identifier;}
    void setIdentifier(std::string i) {identifier = i;}
    virtual void print(){std::cout << "cuobjdump Section: unknown type" << std::endl;}
private:
    unsigned arch;
    std::string identifier;
};

class cuobjdumpELFSection : public cuobjdumpSection
{
public:
    cuobjdumpELFSection() {}
    virtual ~cuobjdumpELFSection() {
        elffilename = "";
        sassfilename = "";
    }    
    std::string getELFfilename() {return elffilename;}
    void setELFfilename(std::string f) {elffilename = f;}
    std::string getSASSfilename() {return sassfilename;}
    void setSASSfilename(std::string f) {sassfilename = f;}
    virtual void print() {
        std::cout << "ELF Section:" << std::endl;
        std::cout << "arch: sm_" << getArch() << std::endl;
        std::cout << "identifier: " << getIdentifier() << std::endl;
        std::cout << "elf filename: " << getELFfilename() << std::endl;
        std::cout << "sass filename: " << getSASSfilename() << std::endl;
        std::cout << std::endl;
    }
private:
    std::string elffilename;
    std::string sassfilename;
};


class cuobjdumpPTXSection : public cuobjdumpSection
{
public:
    cuobjdumpPTXSection(){
        ptxfilename = "";
    }
    std::string getPTXfilename() {return ptxfilename;}
    void setPTXfilename(std::string f) {ptxfilename = f;}
    virtual void print() {
        std::cout << "PTX Section:" << std::endl;
        std::cout << "arch: sm_" << getArch() << std::endl;
        std::cout << "identifier: " << getIdentifier() << std::endl;
        std::cout << "ptx filename: " << getPTXfilename() << std::endl;
        std::cout << std::endl;
    }
private:
    std::string ptxfilename;
};

std::list<cuobjdumpSection*> cuobjdumpSectionList;
std::list<cuobjdumpSection*> libSectionList;

// sectiontype: 0 for ptx, 1 for elf
void addCuobjdumpSection(int sectiontype){
    if (sectiontype)
        cuobjdumpSectionList.push_front(new cuobjdumpELFSection());
    else
        cuobjdumpSectionList.push_front(new cuobjdumpPTXSection());
    printf("## Adding new section %s\n", sectiontype?"ELF":"PTX");
}

void setCuobjdumparch(const char* arch){
    unsigned archnum;
    sscanf(arch, "sm_%u", &archnum);
    assert (archnum && "cannot have sm_0");
    printf("Adding arch: %s\n", arch);
    cuobjdumpSectionList.front()->setArch(archnum);
}

void setCuobjdumpidentifier(const char* identifier){
    printf("Adding identifier: %s\n", identifier);
    cuobjdumpSectionList.front()->setIdentifier(identifier);
}

void setCuobjdumpptxfilename(const char* filename){
    printf("Adding ptx filename: %s\n", filename);
    cuobjdumpSection* x = cuobjdumpSectionList.front();
    if (dynamic_cast<cuobjdumpPTXSection*>(x) == NULL){
        assert (0 && "You shouldn't be trying to add a ptxfilename to an elf section");
    }
    (dynamic_cast<cuobjdumpPTXSection*>(x))->setPTXfilename(filename);
}

void setCuobjdumpelffilename(const char* filename){
    if (dynamic_cast<cuobjdumpELFSection*>(cuobjdumpSectionList.front()) == NULL){
        assert (0 && "You shouldn't be trying to add a elffilename to an ptx section");
    }
    (dynamic_cast<cuobjdumpELFSection*>(cuobjdumpSectionList.front()))->setELFfilename(filename);
}

void setCuobjdumpsassfilename(const char* filename){
    if (dynamic_cast<cuobjdumpELFSection*>(cuobjdumpSectionList.front()) == NULL){
        assert (0 && "You shouldn't be trying to add a sassfilename to an ptx section");
    }
    (dynamic_cast<cuobjdumpELFSection*>(cuobjdumpSectionList.front()))->setSASSfilename(filename);
}
extern int cuobjdump_parse();
extern FILE *cuobjdump_in;



std::string get_app_binary(){
   char self_exe_path[1025];
#ifdef __APPLE__
   uint32_t size = sizeof(self_exe_path);
   if( _NSGetExecutablePath(self_exe_path,&size) != 0 ) {
       printf("GPGPU-Sim ** ERROR: _NSGetExecutablePath input buffer too small\n");
       exit(1);
   }
#else
   std::stringstream exec_link;
   exec_link << "/proc/self/exe";

   ssize_t path_length = readlink(exec_link.str().c_str(), self_exe_path, 1024);
   assert(path_length != -1);
   self_exe_path[path_length] = '\0';
#endif

   printf("self exe links to: %s\n", self_exe_path);
   return self_exe_path;
}

//! Call cuobjdump to extract everything (-elf -sass -ptx)
/*!
 *  This Function extract the whole PTX (for all the files) using cuobjdump
 *  to _cuobjdump_complete_output_XXXXXX then runs a parser to chop it up with each binary in
 *  its own file
 *  It is also responsible for extracting the libraries linked to the binary if the option is
 *  enabled
 * */
void extract_code_using_cuobjdump(){
    //CUctx_st *context = GPGPUSim_Context();
    char command[1000];

   std::string app_binary = get_app_binary();

    char fname[1024];
    snprintf(fname,1024,"_cuobjdump_complete_output_XXXXXX");
    int fd=mkstemp(fname);
    close(fd);
    // Running cuobjdump using dynamic link to current process
    snprintf(command,1000,"md5sum %s ", app_binary.c_str());
    printf("Running md5sum using \"%s\"\n", command);
    system(command);
    // Running cuobjdump using dynamic link to current process
    snprintf(command,1000,"$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass %s > %s", app_binary.c_str(), fname);
    printf("Running cuobjdump using \"%s\"\n", command);
    bool parse_output = true;
    int result = system(command);
    if(result) {
		printf("KAIN Error in system command\n");
		exit(0);
    }

    if (parse_output) {
        printf("Parsing file %s\n", fname);
        cuobjdump_in = fopen(fname, "r");

        cuobjdump_parse();
        fclose(cuobjdump_in);
        printf("Done parsing!!!\n");
    } else {
        printf("Parsing skipped for %s\n", fname);
    }

    if (0){
        //Experimental library support
        //Currently only for cufft

        std::stringstream cmd;
        cmd << "ldd " << app_binary << " | grep $CUDA_INSTALL_PATH | awk \'{print $3}\' > _tempfile_.txt";
        int result = system(cmd.str().c_str());
        if(result){
            //KAIN
            //std::cout << "Failed to execute: " << cmd << std::endl;
            exit(1);
        }
        std::ifstream libsf;
        libsf.open("_tempfile_.txt");
        if(!libsf.is_open()) {
            std::cout << "Failed to open: _tempfile_.txt" << std::endl;
            exit(1);
        }

        //Save the original section list
        std::list<cuobjdumpSection*> tmpsl = cuobjdumpSectionList;
        cuobjdumpSectionList.clear();

        std::string line;
        std::getline(libsf, line);
        std::cout << "DOING: " << line << std::endl;
        int cnt=1;
        while(libsf.good()){
            std::stringstream libcodfn;
            libcodfn << "_cuobjdump_complete_lib_" << cnt << "_";
            cmd.str(""); //resetting
            cmd << "$CUDA_INSTALL_PATH/bin/cuobjdump -ptx -elf -sass ";
            cmd << line;
            cmd << " > ";
            cmd << libcodfn.str();
            std::cout << "Running cuobjdump on " << line << std::endl;
            std::cout << "Using command: " << cmd.str() << std::endl;
            result = system(cmd.str().c_str());
            if(result) {printf("ERROR: Failed to execute: %s\n", command); exit(1);}
            std::cout << "Done" << std::endl;
            //KAIN
            //std::cout << "Trying to parse " << libcodfn << std::endl;
            cuobjdump_in = fopen(libcodfn.str().c_str(), "r");
            cuobjdump_parse();
            fclose(cuobjdump_in);
            std::getline(libsf, line);
        }
        libSectionList = cuobjdumpSectionList;
        cuobjdumpSectionList = tmpsl;
    }
}

char* readfile (const std::string filename){
    assert (filename != "");
    FILE* fp = fopen(filename.c_str(),"r");
    if (!fp) {
        std::cout << "ERROR: Could not open file %s for reading\n" << filename << std::endl;
        assert (0);
    }
    // finding size of the file
    int filesize= 0;
    fseek (fp , 0 , SEEK_END);

    filesize = ftell (fp);
    fseek (fp, 0, SEEK_SET);
    // allocate and copy the entire ptx
    char* ret = (char*)malloc((filesize +1)* sizeof(char));
    fread(ret,1,filesize,fp);
    ret[filesize]='\0';
    fclose(fp);
    return ret;
}

//! Function that helps debugging
void printSectionList(std::list<cuobjdumpSection*> sl) {
    std::list<cuobjdumpSection*>::iterator iter;
    for (   iter = sl.begin();
            iter != sl.end();
            iter++
    ){
        (*iter)->print();
    }
}

std::list<cuobjdumpSection*> pruneSectionList(std::list<cuobjdumpSection*> cuobjdumpSectionList,  unsigned forced_max_capability) {

    //For ptxplus, force the max capability to 19 if it's higher or unspecified(0)

    std::list<cuobjdumpSection*> prunedList;

    //Find the highest capability (that is lower than the forces maximum) for each cubin file
    //and set it in cuobjdumpSectionMap. Do this only for ptx sections
    std::map<std::string, unsigned> cuobjdumpSectionMap;
    for (   std::list<cuobjdumpSection*>::iterator iter = cuobjdumpSectionList.begin();
            iter != cuobjdumpSectionList.end();
            iter++){
        unsigned capability = (*iter)->getArch();
        if(dynamic_cast<cuobjdumpPTXSection*>(*iter) != NULL &&
                (capability <= forced_max_capability ||
                        forced_max_capability==0)) {
            if(cuobjdumpSectionMap[(*iter)->getIdentifier()] < capability)
                cuobjdumpSectionMap[(*iter)->getIdentifier()] = capability;
        }
    }

    //Throw away the sections with the lower capabilites and push those with the highest in
    //the pruned list
    for (   std::list<cuobjdumpSection*>::iterator iter = cuobjdumpSectionList.begin();
            iter != cuobjdumpSectionList.end();
            iter++){
        unsigned capability = (*iter)->getArch();
        if(capability == cuobjdumpSectionMap[(*iter)->getIdentifier()]){
            prunedList.push_back(*iter);
        } else {
            delete *iter;
        }
    }
    return prunedList;
}

//! Within the section list, find the ELF section corresponding to a given identifier
cuobjdumpELFSection* findELFSectionInList(std::list<cuobjdumpSection*> sectionlist, const std::string identifier){

    std::list<cuobjdumpSection*>::iterator iter;
    for (   iter = sectionlist.begin();
            iter != sectionlist.end();
            iter++
    ){
        cuobjdumpELFSection* elfsection;
        if((elfsection=dynamic_cast<cuobjdumpELFSection*>(*iter)) != NULL){
            if(elfsection->getIdentifier() == identifier)
                return elfsection;
        }
    }
    return NULL;
}

//! Find an ELF section in all the known lists
cuobjdumpELFSection* findELFSection(const std::string identifier){
    cuobjdumpELFSection* sec = findELFSectionInList(cuobjdumpSectionList, identifier);
    if (sec!=NULL)return sec;
    sec = findELFSectionInList(libSectionList, identifier);
    if (sec!=NULL)return sec;
    std::cout << "Cound not find " << identifier << std::endl;
    assert(0 && "Could not find the required ELF section");
    return NULL;
}

//! Within the section list, find the PTX section corresponding to a given identifier
cuobjdumpPTXSection* findPTXSectionInList(std::list<cuobjdumpSection*> sectionlist, const std::string identifier){
    std::list<cuobjdumpSection*>::iterator iter;
    for (   iter = sectionlist.begin();
            iter != sectionlist.end();
            iter++
    ){
        cuobjdumpPTXSection* ptxsection;
        if((ptxsection=dynamic_cast<cuobjdumpPTXSection*>(*iter)) != NULL){
            if(ptxsection->getIdentifier() == identifier)
                return ptxsection;
        }
    }
    return NULL;
}

//! Find an PTX section in all the known lists
cuobjdumpPTXSection* findPTXSection(const std::string identifier){
    cuobjdumpPTXSection* sec = findPTXSectionInList(cuobjdumpSectionList, identifier);
    if (sec!=NULL)return sec;
    sec = findPTXSectionInList(libSectionList, identifier);
    if (sec!=NULL)return sec;
    std::cout << "Cound not find " << identifier << std::endl;
    assert(0 && "Could not find the required PTX section");
    return NULL;
}



//! Extract the code using cuobjdump and remove unnecessary sections
void cuobjdumpInit(unsigned forced_max_capability){
    extract_code_using_cuobjdump(); //extract all the output of cuobjdump to _cuobjdump_*.*
    cuobjdumpSectionList = pruneSectionList(cuobjdumpSectionList, forced_max_capability);
}

std::map<int, std::string> fatbinmap;
std::map<int, bool>fatbin_registered;

void cuobjdumpRegisterFatBinary(unsigned int handle, char* filename){
    fatbinmap[handle] = filename;
}













void** CUDARTAPI __cudaRegisterFatBinary( void *fatCubin )
{
  PRINT_CALL;
  // this is always called first for CUDA binary
  Pipe_Init();

#if (CUDART_VERSION < 2010)
  printf("GPGPU-Sim PTX: ERROR ** this version of GPGPU-Sim requires CUDA 2.1 or higher\n");
  exit(1);
#endif
/*
  __cudaFatCudaBinary *info =   (__cudaFatCudaBinary *)fatCubin;
  assert( info->version >= 3 );
  if (!info->ptx){
    printf("ERROR: Cannot find ptx code in cubin file\n"
        "\tIf you are using CUDA 4.0 or higher, please enable -gpgpu_ptx_use_cuobjdump or downgrade to CUDA 3.1\n");
    exit(1);
  }
*/
  unsigned forced_max_capability;
  g_pipe->write(MESSAGE_GPU_REGISTER_FAT_BINARY);
  g_pipe->read(forced_max_capability);

	printf("KAIN: pid %d, read forced_max_capability\n",getpid());
	fflush(stdout);
  bool found=false;
/*
  unsigned num_ptx_versions=0;
  unsigned max_capability=0;
  unsigned selected_capability=0;
  while( info->ptx[num_ptx_versions].gpuProfileName != NULL ) {
    unsigned capability=0;
    sscanf(info->ptx[num_ptx_versions].gpuProfileName,"compute_%u",&capability);
    printf("GPGPU-Sim PTX: __cudaRegisterFatBinary found PTX versions for '%s', ", info->ident);
    printf("capability = %s\n", info->ptx[num_ptx_versions].gpuProfileName );



	printf("KAIN: pid %d, come here\n",getpid());
	fflush(stdout);

    if( forced_max_capability ) {
      if( capability > max_capability && capability <= forced_max_capability ) {
        found = true;
        max_capability=capability;
        selected_capability = num_ptx_versions;
      }
    } else {
      if( capability > max_capability ) {
        found = true;
        max_capability=capability;
        selected_capability = num_ptx_versions;
      }
    }
    num_ptx_versions++;
  }
*/
 

  unsigned long long fat_cubin_handle;
  g_pipe->read(fat_cubin_handle);
    char *ptxcode;

  {
         // FatBin handle from the .fatbin.c file (one of the intermediate files generated by NVCC)
        typedef struct {int m; int v; const unsigned long long* d; char* f;} __fatDeviceText __attribute__ ((aligned (8))); 
        __fatDeviceText * fatDeviceText = (__fatDeviceText *) fatCubin;

        // Extract the source code file name that generate the given FatBin. 
        // - Obtains the pointer to the actual fatbin structure from the FatBin handle (fatCubin).
        // - An integer inside the fatbin structure contains the relative offset to the source code file name.
        // - This offset differs among different CUDA and GCC versions. 
        char * pfatbin = (char*) fatDeviceText->d; 
        int offset = *((int*)(pfatbin+48)); 
        char * filename = (pfatbin+16+offset); 

        // The extracted file name is associated with a fat_cubin_handle passed
        // into cudaLaunch().  Inside cudaLaunch(), the associated file name is
        // used to find the PTX/SASS section from cuobjdump, which contains the
        // PTX/SASS code for the launched kernel function.  
        // This allows us to work around the fact that cuobjdump only outputs the
        // file name associated with each section. 
        printf("GPGPU-Sim PTX: __cudaRegisterFatBinary, fat_cubin_handle = %llu, filename=%s\n", fat_cubin_handle, filename);
        /*!  
         * This function extracts all data from all files in first call
         * then for next calls, only returns the appropriate number
         */
        assert(fat_cubin_handle >= 1);
        if (1==1) cuobjdumpInit(forced_max_capability);//KAIN: as cannot know this is the first time or not, so dump each time
        cuobjdumpRegisterFatBinary(fat_cubin_handle, filename);


//    CUctx_st *context = GPGPUSim_Context();

    std::string fname = fatbinmap[fat_cubin_handle];
    cuobjdumpPTXSection* ptx = findPTXSection(fname);

    const char *override_ptx_name = getenv("PTX_SIM_KERNELFILE"); 

   if (override_ptx_name == NULL or getenv("PTX_SIM_USE_PTX_FILE") == NULL) {
        ptxcode = readfile(ptx->getPTXfilename());
    } else {
        printf("GPGPU-Sim PTX: overriding embedded ptx with '%s' (PTX_SIM_USE_PTX_FILE is set)\n", override_ptx_name);
        ptxcode = readfile(override_ptx_name);
    }


    found = true;
  }

  g_pipe->write(found);

  if( found  ) {
//    printf("GPGPU-Sim PTX: Loading PTX for %s, capability = %s\n",
  //      info->ident, info->ptx[selected_capability].gpuProfileName );

 //   const char *ptx = info->ptx[selected_capability].ptx;
    size_t length = strlen(ptxcode) + 1;
    g_pipe->write(length);
    g_pipe->write(ptxcode, length);

  } else {
    printf("GPGPU-Sim PTX: warning -- did not find an appropriate PTX in cubin\n");
	assert(0);
  }
  return (void**)fat_cubin_handle;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
  PRINT_CALL;

  g_pipe->write(MESSAGE_GPU_EXIT_SIMULATION);

  bool exit_simulation;
  g_pipe->read(exit_simulation);
  if (exit_simulation) {
    printf("GPGPU-Sim Child: Execution finished!\n");
    Pipe_Finish();
  } else {
    printf("GPGPU-Sim Child: Fat binary remaining!\n");
  }
}

cudaError_t cudaDeviceReset ( void ) {
  PRINT_CALL;
  // Should reset the simulated GPU
  return g_last_cudaError = cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceSynchronize(void){
  PRINT_CALL;
  // I don't know what this should do
  return g_last_cudaError = cudaSuccess;
}

void CUDARTAPI __cudaRegisterFunction(
		void   **fatCubinHandle,
		const char    *hostFun,
		char    *deviceFun,
		const char    *deviceName,
		int      thread_limit,
		uint3   *tid,
		uint3   *bid,
		dim3    *bDim,
		dim3    *gDim
)
{
  PRINT_CALL;
  unsigned fat_cubin_handle = (unsigned)(unsigned long long)fatCubinHandle;

  g_pipe->write(MESSAGE_GPU_REGISTER_FUNCTION);
  g_pipe->write(fat_cubin_handle);
  // hostFun pointer itself is used as a key
  g_pipe->write(hostFun);
  // deviceFun string itself is important
  size_t length = strlen(deviceFun) + 1;
  g_pipe->write(length);
  g_pipe->write(deviceFun, length);
}

extern void __cudaRegisterVar(
		void **fatCubinHandle,
		char *hostVar, //pointer to...something
		char *deviceAddress, //name of variable
		const char *deviceName, //name of variable (same as above)
		int ext,
		int size,
		int constant,
		int global )
{
  PRINT_CALL;
  printf("GPGPU-Sim PTX: __cudaRegisterVar: hostVar = %p; deviceAddress = %p; deviceName = %s\n", hostVar, deviceAddress, deviceName);
  printf("GPGPU-Sim PTX: __cudaRegisterVar: Registering const memory space of %d bytes\n", size);
  fflush(stdout);

  if ( constant && !global && !ext ) {
    g_pipe->write(MESSAGE_GPU_REGISTER_CONST_VARIABLE);
    //gpgpu_ptx_sim_register_const_variable(hostVar,deviceName,size);
  } else if ( !constant && !global && !ext ) {
    g_pipe->write(MESSAGE_GPU_REGISTER_GLOBAL_VARIABLE);
    //gpgpu_ptx_sim_register_global_variable(hostVar,deviceName,size);
  } else {
    cuda_not_implemented(__my_func__,__LINE__);
  }

  // hostVar pointer itself is used as a key
  g_pipe->write(hostVar);
  size_t hostVar_length = strlen(hostVar) + 1;
  g_pipe->write(hostVar_length);
  g_pipe->write(hostVar, hostVar_length);
  // deviceName is important
  size_t deviceName_length = strlen(deviceName) + 1;
  g_pipe->write(deviceName_length);
  g_pipe->write(deviceName, deviceName_length);
  // size
  g_pipe->write(size);
}


void __cudaRegisterShared(
		void **fatCubinHandle,
		void **devicePtr
)
{
  PRINT_CALL;
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterShared\n" );
}

void CUDARTAPI __cudaRegisterSharedVar(
		void   **fatCubinHandle,
		void   **devicePtr,
		size_t   size,
		size_t   alignment,
		int      storage
)
{
  PRINT_CALL;
  // we don't do anything here
  printf("GPGPU-Sim PTX: __cudaRegisterSharedVar\n" );
}

void __cudaRegisterTexture(
		void **fatCubinHandle,
		const struct textureReference *hostVar,
		const void **deviceAddress,
		const char *deviceName,
		int dim,
		int norm,
		int ext
) //passes in a newly created textureReference
{
  PRINT_CALL;

  g_pipe->write(MESSAGE_GPU_REGISTER_TEXTURE);
  // both pointer itself and the content of hostVar is important
  g_pipe->write(hostVar);
  g_pipe->write(*hostVar);
  // deviceName is explicitly copied
  size_t length = strlen(deviceName) + 1;
  g_pipe->write(length);
  g_pipe->write(deviceName, length);

  g_pipe->write(dim);
  g_pipe->write(norm);
  g_pipe->write(ext);
}

#ifndef OPENGL_SUPPORT
typedef unsigned long GLuint;
#endif

cudaError_t cudaGLRegisterBufferObject(GLuint bufferObj)
{
  PRINT_CALL;
  printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
  return g_last_cudaError = cudaSuccess;
}

struct glbmap_entry {
	GLuint m_bufferObj;
	void *m_devPtr;
	size_t m_size;
	struct glbmap_entry *m_next;
};
typedef struct glbmap_entry glbmap_entry_t;

glbmap_entry_t* g_glbmap = NULL;

cudaError_t cudaGLMapBufferObject(void** devPtr, GLuint bufferObj) 
{
  PRINT_CALL;
  /*
#ifdef OPENGL_SUPPORT
	GLint buffer_size=0;
	CUctx_st* ctx = GPGPUSim_Context();

	glbmap_entry_t *p = g_glbmap;
	while ( p && p->m_bufferObj != bufferObj )
		p = p->m_next;
	if ( p == NULL ) {
		glBindBuffer(GL_ARRAY_BUFFER,bufferObj);
		glGetBufferParameteriv(GL_ARRAY_BUFFER,GL_BUFFER_SIZE,&buffer_size);
		assert( buffer_size != 0 );
		*devPtr = ctx->get_device()->get_gpgpu()->gpu_malloc(buffer_size);

		// create entry and insert to front of list
		glbmap_entry_t *n = (glbmap_entry_t *) calloc(1,sizeof(glbmap_entry_t));
		n->m_next = g_glbmap;
		g_glbmap = n;

		// initialize entry
		n->m_bufferObj = bufferObj;
		n->m_devPtr = *devPtr;
		n->m_size = buffer_size;

		p = n;
	} else {
		buffer_size = p->m_size;
		*devPtr = p->m_devPtr;
	}

	if ( *devPtr  ) {
		char *data = (char *) calloc(p->m_size,1);
		glGetBufferSubData(GL_ARRAY_BUFFER,0,buffer_size,data);
		memcpy_to_gpu( (size_t) *devPtr, data, buffer_size );
		free(data);
		printf("GPGPU-Sim PTX: cudaGLMapBufferObject %zu bytes starting at 0x%llx..\n", (size_t)buffer_size,
				(unsigned long long) *devPtr);
		return g_last_cudaError = cudaSuccess;
	} else {
		return g_last_cudaError = cudaErrorMemoryAllocation;
	}

	return g_last_cudaError = cudaSuccess;
#else
	fflush(stdout);
	fflush(stderr);
	printf("GPGPU-Sim PTX: GPGPU-Sim support for OpenGL integration disabled -- exiting\n");
	fflush(stdout);
	exit(50);
#endif
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t cudaGLUnmapBufferObject(GLuint bufferObj)
{
  PRINT_CALL;
  /*
#ifdef OPENGL_SUPPORT
	glbmap_entry_t *p = g_glbmap;
	while ( p && p->m_bufferObj != bufferObj )
		p = p->m_next;
	if ( p == NULL )
		return g_last_cudaError = cudaErrorUnknown;

	char *data = (char *) calloc(p->m_size,1);
	memcpy_from_gpu( data,(size_t)p->m_devPtr,p->m_size );
	glBufferSubData(GL_ARRAY_BUFFER,0,p->m_size,data);
	free(data);

	return g_last_cudaError = cudaSuccess;
#else
	fflush(stdout);
	fflush(stderr);
	printf("GPGPU-Sim PTX: support for OpenGL integration disabled -- exiting\n");
	fflush(stdout);
	exit(50);
#endif
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t cudaGLUnregisterBufferObject(GLuint bufferObj) 
{
  PRINT_CALL;
  /*
	printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

#if (CUDART_VERSION >= 2010)

cudaError_t CUDARTAPI cudaHostAlloc(void **pHost,  size_t bytes, unsigned int flags)
{
  PRINT_CALL;
  *pHost = malloc(bytes);
  if( *pHost )
    return g_last_cudaError = cudaSuccess;
  else
    return g_last_cudaError = cudaErrorMemoryAllocation;
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaSetDeviceFlags( int flags )
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const char *hostFun )
{
  PRINT_CALL;
  /*
	CUctx_st *context = GPGPUSim_Context();
	function_info *entry = context->get_kernel(hostFun);
	if( entry ) {
		const struct gpgpu_ptx_sim_kernel_info *kinfo = entry->get_kernel_info();
		attr->sharedSizeBytes = kinfo->smem;
		attr->constSizeBytes  = kinfo->cmem;
		attr->localSizeBytes  = kinfo->lmem;
		attr->numRegs         = kinfo->regs;
		attr->maxThreadsPerBlock = 0; // from pragmas?
#if CUDART_VERSION >= 3000
		attr->ptxVersion      = kinfo->ptx_version;
		attr->binaryVersion   = kinfo->sm_target;
#endif
	}
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, int flags)
{
  PRINT_CALL;
  /*
	CUevent_st *e = new CUevent_st(flags==cudaEventBlockingSync);
	g_timer_events[e->get_uid()] = e;
#if CUDART_VERSION >= 3000
	*event = e;
#else
	*event = e->get_uid();
#endif
	return g_last_cudaError = cudaSuccess;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion)
{
  PRINT_CALL;
  /*
  *driverVersion = CUDART_VERSION;
  return g_last_cudaError = cudaErrorUnknown;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

cudaError_t CUDARTAPI cudaRuntimeGetVersion(int *runtimeVersion)
{
  PRINT_CALL;
  /*
  *runtimeVersion = CUDART_VERSION;
  return g_last_cudaError = cudaErrorUnknown;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

#if CUDART_VERSION >= 3000
__host__ cudaError_t CUDARTAPI cudaFuncSetCacheConfig(const char *func, enum cudaFuncCache  cacheConfig )
{
  PRINT_CALL;

  g_pipe->write(MESSAGE_GPU_SET_CACHE_CONFIG);
  // pointer itself is important
  g_pipe->write(func);
  g_pipe->write(cacheConfig);
  return g_last_cudaError = cudaSuccess;
}
#endif

#endif

cudaError_t CUDARTAPI cudaGLSetGLDevice(int device)
{
  PRINT_CALL;
  /*
	printf("GPGPU-Sim PTX: Execution warning: ignoring call to \"%s\"\n", __my_func__ );
	return g_last_cudaError = cudaErrorUnknown;
  */
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

typedef void* HGPUNV;

cudaError_t CUDARTAPI cudaWGLGetDevice(int *device, HGPUNV hGpu)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
  return g_last_cudaError = cudaErrorUnknown;
}

void CUDARTAPI __cudaMutexOperation(int lock)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
}

void  CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val) 
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
}

}

namespace cuda_math {

void CUDARTAPI __cudaMutexOperation(int lock)
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
}

void  CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val) 
{
  PRINT_CALL;
  cuda_not_implemented(__my_func__,__LINE__);
}

int CUDARTAPI __cudaSynchronizeThreads(void**, void*)
{
  PRINT_CALL;
  //TODO This function should syncronize if we support Asyn kernel calls
  return g_last_cudaError = cudaSuccess;
}

}




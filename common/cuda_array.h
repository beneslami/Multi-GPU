#ifndef __CUDA_ARRAY_H__
#define __CUDA_ARRAY_H__

struct cudaArray {
   void *devPtr;
   int devPtr32;
   struct cudaChannelFormatDesc desc;
   int width;
   int height;
   int size; //in bytes
   unsigned dimensions;
};

#endif // __CUDA_ARRAY_H__


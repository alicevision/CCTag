#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <cuda_runtime.h>

namespace popart {

void pop_info_gridsize( bool               silent,
                        dim3&              grid,
                        dim3&              block,
                        const std::string& kernel,
                        const char*        file,
                        size_t             line );
#define POP_INFO_GRIDSIZE(silent,grid,block,kernel) \
    pop_info_gridsize(silent,grid,block,kernel,__FILE__,__LINE__)

void pop_stream_synchronize( cudaStream_t stream,
                             const char*  file,
                             size_t       line );
#define POP_SYNC( stream ) pop_stream_synchronize( stream, __FILE__, __LINE__ )

void pop_check_last_error( const char* file,
                           size_t      line );
#define POP_CHK pop_check_last_error( __FILE__, __LINE__ )

void pop_cuda_memcpy_async( void*          dst,
                            const void*    src,
                            size_t         sz,
                            cudaMemcpyKind type,
                            cudaStream_t   stream,
                            bool           silent,
                            const char*    file,
                            size_t         line );
#define POP_CUDA_MEMCPY_ASYNC( dst, src, sz, type, stream, silent ) \
    pop_cuda_memcpy_async( dst, src, sz, type, stream, silent, __FILE__, __LINE__ )

void pop_cuda_memcpy( void*          dst,
                      const void*    src,
                      size_t         sz,
                      cudaMemcpyKind type,
                      const char*    file,
                      size_t         line );
#define POP_CUDA_MEMCPY_SYNC( dst, src, sz, type ) \
    pop_cuda_memcpy( dst, src, sz, type, __FILE__, __LINE__ )

void pop_cuda_memcpy_2D( void*          dst,
                         size_t         dpitch,
                         const void*    src,
                         size_t         spitch,
                         size_t         width,
                         size_t         height,
                         cudaMemcpyKind type,
                         const char*    file,
                         size_t         line );
#define POP_CUDA_MEMCPY_2D( dst, dpitch, src, spitch, width, height, type ) \
    pop_cuda_memcpy_2D( dst, dpitch, src, spitch, width, height, type, __FILE__, __LINE__ )

void pop_cuda_memcpy_2D_async( void*          dst,
                               size_t         dpitch,
                               const void*    src,
                               size_t         spitch,
                               size_t         width,
                               size_t         height,
                               cudaMemcpyKind type,
                               cudaStream_t   stream,
                               const char*    file,
                               size_t         line );
#define POP_CUDA_MEMCPY_2D_ASYNC( dst, dpitch, src, spitch, width, height, type, stream ) \
    pop_cuda_memcpy_2D_async( dst, dpitch, src, spitch, width, height, type, stream, __FILE__, __LINE__ )

void pop_cuda_memcpy_to_symbol_sync( const void*    symbol,
                                     const void*    src,
                                     size_t         sz,
                                     size_t         offset,
                                     cudaMemcpyKind type,
                                     const char*    file,
                                     size_t         line );
#define POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( symbol, src, sz ) \
    pop_cuda_memcpy_to_symbol_sync( symbol, src, sz, 0, cudaMemcpyHostToDevice, __FILE__, __LINE__ )

void pop_cuda_memcpy_to_symbol_async( const void*    symbol,
                                      const void*    src,
                                      size_t         sz,
                                      size_t         offset,
                                      cudaMemcpyKind type,
                                      cudaStream_t   stream,
                                      const char*    file,
                                      size_t         line );
#define POP_CUDA_MEMCPY_HOST_TO_SYMBOL_ASYNC( symbol, src, sz, stream ) \
    pop_cuda_memcpy_to_symbol_async( symbol, src, sz, 0, cudaMemcpyHostToDevice, stream, __FILE__, __LINE__ )

void pop_cuda_memset_async( void*        ptr,
                            int          value,
                            size_t       bytes,
                            cudaStream_t stream,
                            const char*  file,
                            size_t       line );
#define POP_CUDA_MEMSET_ASYNC( ptr, val, sz, stream ) \
    pop_cuda_memset_async( ptr, val, sz, stream, __FILE__, __LINE__ )

void pop_cuda_memset( void*        ptr,
                      int          value,
                      size_t       bytes,
                      const char*  file,
                      size_t       line );
#define POP_CUDA_MEMSET_SYNC( ptr, val, sz ) \
    pop_cuda_memset( ptr, val, sz, __FILE__, __LINE__ )

#define POP_FATAL(s) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << std::endl; \
        exit( -__LINE__ ); \
    }

#define POP_FATAL_FL(s,file,line) { \
        std::cerr << file << ":" << line << std::endl << "    " << s << std::endl; \
        exit( -__LINE__ ); \
    }

#define POP_CHECK_NON_NULL(ptr,s) if( ptr == 0 ) { POP_FATAL(s); }

#define POP_CHECK_NON_NULL_FL(ptr,s,file,line) if( ptr == 0 ) { POP_FATAL_FL(s,file,line); }

#define POP_INFO(s)
// #define POP_INFO(s) cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << endl

#define POP_INFO2(silent,s) \
    if (not silent) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << std::endl; \
    }

#define POP_CUDA_FATAL_FL(err,s,file,line) { \
        std::cerr << file << ":" << line << std::endl; \
        std::cerr << "    " << s << cudaGetErrorString(err) << std::endl; \
        exit( -__LINE__ ); \
    }
#define POP_CUDA_FATAL(err,s) POP_CUDA_FATAL_FL(err,s,__FILE__,__LINE__)
#define POP_CUDA_FATAL_TEST_FL(err,s,file,line) if( err != cudaSuccess ) { POP_CUDA_FATAL_FL(err,s,file,line); }
#define POP_CUDA_FATAL_TEST(err,s) if( err != cudaSuccess ) { POP_CUDA_FATAL(err,s); }

#if 0
#define POP_CUDA_MALLOC_INIT0( ptr, sz ) { \
        cudaError_t err; \
        err = cudaMalloc( ptr, sz ); \
        POP_CUDA_FATAL_TEST( err, "cudaMalloc failed: " ); \
        err = cudaMemset( *ptr, 0, sz ); \
        POP_CUDA_FATAL_TEST( err, "cudaMemset failed: " ); \
    }

#define POP_CUDA_MALLOC_NOINIT( ptr, sz ) { \
        cudaError_t err; \
        err = cudaMalloc( ptr, sz ); \
        POP_CUDA_FATAL_TEST( err, "cudaMalloc failed: " ); \
    }
#endif

void pop_cuda_malloc( void** ptr,  uint32_t byte_size, const char* file, uint32_t line );
#define POP_CUDA_MALLOC( ptr, byte_size ) \
    pop_cuda_malloc( ptr, byte_size, __FILE__, __LINE__ )

void pop_cuda_malloc_pitch( void** ptr, size_t* byte_pitch, uint32_t byte_width, uint32_t byte_height, const char* file, uint32_t line );
#define POP_CUDA_MALLOC_PITCH( ptr, byte_pitch, byte_width, byte_height ) \
    pop_cuda_malloc_pitch( ptr, byte_pitch, byte_width, byte_height, __FILE__, __LINE__ )

#define POP_CUDA_MALLOC_HOST( ptr, sz ) { \
        cudaError_t err; \
        err = cudaMallocHost( ptr, sz ); \
        POP_CUDA_FATAL_TEST( err, "cudaMallocHost failed: " ); \
    }

void pop_cuda_free( void* ptr, const char* file, uint32_t line );
#define POP_CUDA_FREE( ptr ) \
    pop_cuda_free( ptr, __FILE__, __LINE__ )

void pop_cuda_stream_create( cudaStream_t* stream, const char* file, uint32_t line );
#define POP_CUDA_STREAM_CREATE( stream ) \
    pop_cuda_stream_create( stream, __FILE__, __LINE__ )

void pop_cuda_stream_destroy( cudaStream_t stream, const char* file, uint32_t line );
#define POP_CUDA_STREAM_DESTROY( stream ) \
    pop_cuda_stream_destroy( stream, __FILE__, __LINE__ )

}; // namespace popart


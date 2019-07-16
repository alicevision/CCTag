/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>

namespace cctag {

/*************************************************************
 * Global switch to use exclusively synchronous CUDA calls
 * for everything using the debug APIs.
 * Terribly slow but helps debugging.
 *************************************************************/

void pop_cuda_only_sync_calls( bool on );

/*************************************************************
 * Group: warning and error messages
 *************************************************************/

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

#define POP_CUDA_FATAL_FL(err,s,file,line) { \
        std::cerr << file << ":" << line << std::endl; \
        std::cerr << "    " << s << cudaGetErrorString(err) << std::endl; \
        exit( -__LINE__ ); \
    }

#define POP_CUDA_FATAL(err,s) POP_CUDA_FATAL_FL(err,s,__FILE__,__LINE__)

#define POP_CUDA_FATAL_TEST_FL(err,s,file,line) if( err != cudaSuccess ) { POP_CUDA_FATAL_FL(err,s,file,line); }

#define POP_CUDA_FATAL_TEST(err,s) if( err != cudaSuccess ) { POP_CUDA_FATAL(err,s); }

#define POP_CHK_CALL_IFSYNC pop_cuda_checkerror_ifsync( __FILE__, __LINE__ )

void pop_cuda_checkerror_ifsync( const char* file, size_t line );

void pop_check_last_error( const char* file,
                           size_t      line );

void pop_sync_and_check_last_error( const char* file,
                                    size_t      line );
#define POP_SYNC_CHK pop_sync_and_check_last_error( __FILE__, __LINE__ )

void pop_info_gridsize( bool               silent,
                        dim3&              grid,
                        dim3&              block,
                        const std::string& kernel,
                        const char*        file,
                        size_t             line );

/*************************************************************
 * Group: memcpy and memset
 *************************************************************/

void pop_cuda_memcpy_async( void*          dst,
                            const void*    src,
                            size_t         sz,
                            cudaMemcpyKind type,
                            cudaStream_t   stream,
                            const char*    file,
                            size_t         line );
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
void pop_cuda_memcpy_to_symbol_async( const void*    symbol,
                                      const void*    src,
                                      size_t         sz,
                                      size_t         offset,
                                      cudaMemcpyKind type,
                                      cudaStream_t   stream,
                                      const char*    file,
                                      size_t         line );
void pop_cuda_memset_async( void*        ptr,
                            int          value,
                            size_t       bytes,
                            cudaStream_t stream,
                            const char*  file,
                            size_t       line );
template <typename T>
inline void pop_cuda_set0_async( T*           ptr,
                                 cudaStream_t stream,
                                 const char*  file,
                                 size_t       line )
{
    pop_cuda_memset_async( (void*)ptr, 0, sizeof(T), stream, file, line );
}
template <typename T>
inline void pop_cuda_setx_async( T*           ptr,
                                 T            val,
                                 cudaStream_t stream,
                                 const char*  file,
                                 size_t       line )
{
    pop_cuda_memcpy_async( (void*)ptr, &val, sizeof(T),
                           cudaMemcpyHostToDevice, stream, file, line );
}

void pop_cuda_memcpy_sync( void*          dst,
                           const void*    src,
                           size_t         sz,
                           cudaMemcpyKind type,
                           const char*    file,
                           size_t         line );
void pop_cuda_memcpy_2D_sync( void*          dst,
                              size_t         dpitch,
                              const void*    src,
                              size_t         spitch,
                              size_t         width,
                              size_t         height,
                              cudaMemcpyKind type,
                              const char*    file,
                              size_t         line );
void pop_cuda_memcpy_to_symbol_sync( const void*    symbol,
                                     const void*    src,
                                     size_t         sz,
                                     size_t         offset,
                                     cudaMemcpyKind type,
                                     const char*    file,
                                     size_t         line );
void pop_cuda_memset_sync( void*        ptr,
                           int          value,
                           size_t       bytes,
                           const char*  file,
                           size_t       line );

/* async */
#define POP_CUDA_MEMCPY_ASYNC( dst, src, sz, type, stream ) \
    pop_cuda_memcpy_async( dst, src, sz, type, stream, __FILE__, __LINE__ )

#define POP_CUDA_MEMCPY_TO_HOST_ASYNC( dst, src, sz, stream ) \
    pop_cuda_memcpy_async( dst, src, sz, cudaMemcpyDeviceToHost, stream, __FILE__, __LINE__ )

#define POP_CUDA_MEMCPY_TO_DEVICE_ASYNC( dst, src, sz, stream ) \
    pop_cuda_memcpy_async( dst, src, sz, cudaMemcpyHostToDevice, stream, __FILE__, __LINE__ )

#define POP_CUDA_MEMCPY_2D_ASYNC( dst, dpitch, src, spitch, width, height, type, stream ) \
    pop_cuda_memcpy_2D_async( dst, dpitch, src, spitch, width, height, type, stream, __FILE__, __LINE__ )

#define POP_CUDA_MEMCPY_HOST_TO_SYMBOL_ASYNC( symbol, src, sz, stream ) \
    pop_cuda_memcpy_to_symbol_async( symbol, src, sz, 0, cudaMemcpyHostToDevice, stream, __FILE__, __LINE__ )

#define POP_CUDA_MEMSET_ASYNC( ptr, val, sz, stream ) \
    pop_cuda_memset_async( ptr, val, sz, stream, __FILE__, __LINE__ )

#define POP_CUDA_SET0_ASYNC( ptr, stream ) \
    pop_cuda_set0_async( ptr, stream, __FILE__, __LINE__ )

#define POP_CUDA_SETX_ASYNC( ptr, x, stream ) \
    pop_cuda_setx_async( ptr, x, stream, __FILE__, __LINE__ )

/* sync */
#define POP_CUDA_MEMCPY_SYNC( dst, src, sz, type ) \
    pop_cuda_memcpy_sync( dst, src, sz, type, __FILE__, __LINE__ )

#define POP_CUDA_MEMCPY_TO_HOST_SYNC( dst, src, sz ) \
    pop_cuda_memcpy_sync( dst, src, sz, cudaMemcpyDeviceToHost, __FILE__, __LINE__ )

#define POP_CUDA_MEMCPY_TO_DEVICE_SYNC( dst, src, sz ) \
    pop_cuda_memcpy_sync( dst, src, sz, cudaMemcpyHostToDevice, __FILE__, __LINE__ )

#define POP_CUDA_MEMCPY_2D_SYNC( dst, dpitch, src, spitch, width, height, type ) \
    pop_cuda_memcpy_2D_sync( dst, dpitch, src, spitch, width, height, type, __FILE__, __LINE__ )

#define POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( symbol, src, sz ) \
    pop_cuda_memcpy_to_symbol_sync( symbol, src, sz, 0, cudaMemcpyHostToDevice, __FILE__, __LINE__ )

#define POP_CUDA_MEMSET_SYNC( ptr, val, sz ) \
    pop_cuda_memset_sync( ptr, val, sz, __FILE__, __LINE__ )

/*************************************************************
 * Group: memory allocation and release
 *************************************************************/

void pop_cuda_malloc( void**      ptr,
                      uint32_t    byte_size,
                      const char* file, uint32_t line );
void pop_cuda_malloc_pitch( void**      ptr,
                            size_t*     byte_pitch,
                            uint32_t    byte_width, uint32_t byte_height,
                            const char* file, uint32_t line );
void pop_cuda_free( void* ptr, const char* file, uint32_t line );
void pop_cuda_free_host( void* ptr, const char* file, uint32_t line );

#define POP_CUDA_MALLOC( ptr, byte_size ) \
    pop_cuda_malloc( ptr, byte_size, __FILE__, __LINE__ )

#define POP_CUDA_MALLOC_INIT0( ptr, sz ) { \
        POP_CUDA_MALLOC( ptr, sz ); \
        POP_CUDA_MEMSET_SYNC( ptr, 0, sz ); \
    }

#define POP_CUDA_MALLOC_NOINIT( ptr, sz ) \
    POP_CUDA_MALLOC( ptr, sz )

#define POP_CUDA_MALLOC_PITCH( ptr, byte_pitch, byte_width, byte_height ) \
    pop_cuda_malloc_pitch( ptr, byte_pitch, byte_width, byte_height, __FILE__, __LINE__ )

#define POP_CUDA_MALLOC_HOST( ptr, sz ) { \
        cudaError_t err; \
        err = cudaMallocHost( ptr, sz ); \
        POP_CUDA_FATAL_TEST( err, "cudaMallocHost failed: " ); \
    }

#define POP_CUDA_FREE( ptr ) \
    pop_cuda_free( ptr, __FILE__, __LINE__ )

#define POP_CUDA_FREE_HOST( ptr ) \
    pop_cuda_free_host( ptr, __FILE__, __LINE__ )

/*************************************************************
 * Group: CUDA stream handling
 *************************************************************/

void pop_cuda_stream_create( cudaStream_t* stream, const char* file, uint32_t line );
void pop_cuda_stream_destroy( cudaStream_t stream, const char* file, uint32_t line );
void pop_stream_synchronize( cudaStream_t stream,
                             const char*  file,
                             size_t       line );

#define POP_CUDA_STREAM_CREATE( stream ) \
    pop_cuda_stream_create( stream, __FILE__, __LINE__ )

#define POP_CUDA_STREAM_DESTROY( stream ) \
    pop_cuda_stream_destroy( stream, __FILE__, __LINE__ )

#define POP_CUDA_SYNC( stream ) \
    pop_stream_synchronize( stream, __FILE__, __LINE__ )

}; // namespace cctag


/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "debug_macros.hpp"

#include <assert.h>

using namespace std;

namespace cctag {

static bool cuda_only_sync_calls = false;

void pop_cuda_only_sync_calls( bool on )
{
    cuda_only_sync_calls = on;
}

void pop_check_last_error( const char* file, size_t line )
{
    cudaError_t err = cudaGetLastError( );
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << file << ":" << line << std::endl
                  << "    cudaGetLastError failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

void pop_sync_and_check_last_error( const char* file, size_t line )
{
    cudaDeviceSynchronize();
    pop_check_last_error( file, line );
}

void pop_cuda_checkerror_ifsync( const char* file, size_t line )
{
    if( ! cuda_only_sync_calls ) return;

    cudaDeviceSynchronize();
    pop_check_last_error( file, line );
}

void pop_info_gridsize( bool silent, dim3& grid,dim3& block, const string& kernel, const char* file, size_t line )
{
    if( silent ) return;

//    std::cerr << __FILE__ << ":" << __LINE__ << std::endl
//              << "    called from " << file << ":" << line << std::endl;
    std::cerr << "    " << kernel << " started with "
              << grid.x*grid.y*grid.z*block.x*block.y*block.z
              << " threads (";
    if( grid.z == 1 && grid.y == 1 )
        std::cerr << grid.x;
    else if( grid.z == 1 )
        std::cerr << "{" << grid.x << "," << grid.y << ")";
    else
        std::cerr << "{" << grid.x << "," << grid.y << "," << grid.z << ")";
    std::cerr << " blocks a ";
    if( block.z == 1 && block.y == 1 )
        std::cerr << block.x;
    else if( block.z == 1 )
        std::cerr << "{" << block.x << "," << block.y << ")";
    else
        std::cerr << "{" << block.x << "," << block.y << "," << block.z << ")";
    std::cerr << " threads)"
              << endl;
}

void pop_stream_synchronize( cudaStream_t stream, const char* file, size_t line )
{
    cudaError_t err = cudaStreamSynchronize( stream );
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << file << ":" << line << std::endl
                  << "    cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

void pop_cuda_malloc( void** ptr,  uint32_t byte_size, const char* file, uint32_t line )
{
    cudaError_t err;
    err = cudaMalloc( ptr, byte_size );
    POP_CUDA_FATAL_TEST_FL( err, "cudaMalloc failed to allocate device memory: ", file, line );
#ifndef NDEBUG
    pop_cuda_memset_sync( *ptr, 255, byte_size, file, line );
#endif // NDEBUG
}

void pop_cuda_malloc_pitch( void** ptr, size_t* byte_pitch, uint32_t byte_width, uint32_t byte_height, const char* file, uint32_t line )
{
    cudaError_t err;
    err = cudaMallocPitch( ptr, byte_pitch, byte_width, byte_height );
    POP_CUDA_FATAL_TEST_FL( err, "cudaMallocPitch failed to allocate device memory: ", file, line );
#ifndef NDEBUG
    pop_cuda_memset_sync( *ptr, 255, (*byte_pitch)*byte_height, file, line );
#endif // NDEBUG
}

void pop_cuda_free( void* ptr, const char* file, uint32_t line )
{
    cudaError_t err;
    err = cudaFree( ptr );
    POP_CUDA_FATAL_TEST_FL( err, "cudaFree failed to release device memory: ", file, line );
}

void pop_cuda_free_host( void* ptr, const char* file, uint32_t line )
{
    cudaError_t err;
    err = cudaFreeHost( ptr );
    POP_CUDA_FATAL_TEST_FL( err, "cudaFree failed to release device memory: ", file, line );
}

void pop_cuda_memcpy_async( void* dst, const void* src, size_t sz, cudaMemcpyKind type, cudaStream_t stream, const char* file, size_t line )
{
    if( cuda_only_sync_calls ) {
        pop_cuda_memcpy_sync( dst, src, sz, type, file, line );
        return;
    }

    POP_CHECK_NON_NULL_FL( dst, "Dest ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( src, "Source ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( sz, "Size in memcpy async is null.", file, line );

    cudaError_t err;
    err = cudaMemcpyAsync( dst, src, sz, type, stream );
    if( err != cudaSuccess ) {
        cerr << file << ":" << line << endl
             << "    " << "Failed to copy "
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": ";
        cerr << cudaGetErrorString(err) << endl;
        cerr << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)dst << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void pop_cuda_memcpy_sync( void* dst, const void* src, size_t sz, cudaMemcpyKind type, const char* file, size_t line )
{
    POP_CHECK_NON_NULL_FL( dst, "Dest ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( src, "Source ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( sz, "Size in memcpy async is null.", file, line );

    cudaError_t err;
    err = cudaMemcpy( dst, src, sz, type );
    if( err != cudaSuccess ) {
        cerr << file << ":" << line << endl
             << "    " << "Failed to copy "
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": ";
        cerr << cudaGetErrorString(err) << endl;
        cerr << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)dst << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void pop_cuda_memcpy_2D_sync( void*          dst,
                              size_t         dpitch,
                              const void*    src,
                              size_t         spitch,
                              size_t         width,
                              size_t         height,
                              cudaMemcpyKind type,
                              const char*    file,
                              size_t         line )
{
    cudaError_t err;
    err = cudaMemcpy2D( dst, dpitch, src, spitch, width, height, type );
    if( err != cudaSuccess ) {
        cerr << file << ":" << line << endl
             << "    cudaMemcpy2D failed to copy "
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": "
             << cudaGetErrorString(err) << endl
             << "    src ptr=" << hex << (size_t)src << dec << " src pitch=" << spitch << endl
             << "    dst ptr=" << hex << (size_t)dst << dec << " dst pitch=" << dpitch << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void pop_cuda_memcpy_2D_async( void*          dst,
                               size_t         dpitch,
                               const void*    src,
                               size_t         spitch,
                               size_t         width,
                               size_t         height,
                               cudaMemcpyKind type,
                               cudaStream_t   stream,
                               const char*    file,
                               size_t         line )
{
    if( cuda_only_sync_calls ) {
        pop_cuda_memcpy_2D_sync( dst, dpitch,
                                 src, spitch,
                                 width, height,
                                 type,
                                 file, line );
        return;
    }

    cudaError_t err;
    err = cudaMemcpy2DAsync( dst, dpitch, src, spitch, width, height, type, stream );
    if( err != cudaSuccess ) {
        cerr << file << ":" << line << endl
             << "    cudaMemcpy2DAsync failed to copy "
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": "
             << cudaGetErrorString(err) << endl
             << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)dst << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void pop_cuda_memcpy_to_symbol_async( const void*    symbol,
                                      const void*    src,
                                      size_t         sz,
                                      size_t         offset,
                                      cudaMemcpyKind type,
                                      cudaStream_t   stream,
                                      const char*    file,
                                      size_t         line )
{
    if( cuda_only_sync_calls ) {
        pop_cuda_memcpy_to_symbol_sync( symbol, src, sz, offset, type, file, line );
        return;
    }

    POP_CHECK_NON_NULL( src, "Source ptr in memcpy async is null." );
    POP_CHECK_NON_NULL( sz, "Size in memcpy async is null." );

    cudaError_t err;
    err = cudaMemcpyToSymbolAsync( symbol, src, sz, offset, type, stream );
    if( err != cudaSuccess ) {
        cerr << file << ":" << line << endl
             << "    " << "Failed to copy to symbol "
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": ";
        cerr << cudaGetErrorString(err) << endl;
        cerr << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)symbol << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void pop_cuda_memcpy_to_symbol_sync( const void*    symbol,
                                     const void*    src,
                                     size_t         sz,
                                     size_t         offset,
                                     cudaMemcpyKind type,
                                     const char*    file,
                                     size_t         line )
{
    POP_CHECK_NON_NULL( src, "Source ptr in memcpy async is null." );
    POP_CHECK_NON_NULL( sz, "Size in memcpy async is null." );

    cudaError_t err;
    err = cudaMemcpyToSymbol( symbol, src, sz, offset, type );
    if( err != cudaSuccess ) {
        cerr << file << ":" << line << endl
             << "    " << "Failed to copy to symbol "
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": ";
        cerr << cudaGetErrorString(err) << endl;
        cerr << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)symbol << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void pop_cuda_memset_async( void* ptr, int value, size_t bytes, cudaStream_t stream, const char* file, size_t line )
{
    if( cuda_only_sync_calls ) {
        pop_cuda_memset_sync( ptr, value, bytes, file, line );
        return;
    }

    cudaError_t err;
    err = cudaMemsetAsync( ptr, value, bytes, stream );
    POP_CUDA_FATAL_TEST_FL( err, "cudaMemsetAsync failed: ", file, line );
}

void pop_cuda_memset_sync( void* ptr, int value, size_t bytes, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaMemset( ptr, value, bytes );
    POP_CUDA_FATAL_TEST_FL( err, "cudaMemset failed: ", file, line );
}

void pop_cuda_stream_create( cudaStream_t* stream, const char* file, uint32_t line )
{
    cudaError_t err;
    err = cudaStreamCreate( stream );
    POP_CUDA_FATAL_TEST_FL( err, "cudaStreamCreate failed: ", file, line );
}

void pop_cuda_stream_destroy( cudaStream_t stream, const char* file, uint32_t line )
{
    cudaError_t err;
    err = cudaStreamDestroy( stream );
    POP_CUDA_FATAL_TEST_FL( err, "cudaStreamDestroy failed: ", file, line );
}

}; // namespace cctag


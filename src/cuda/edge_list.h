#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>

#include "triple_point.h"
#include "debug_macros.hpp"

namespace popart {

template <typename T>
struct DevEdgeList
{
    T*   ptr;
    int* size;

    __device__
    int Size() const { return *size; }
};

template <typename T>
struct HostEdgeList
{
    int size;

    HostEdgeList()
        : size(0)
    { }

    ~HostEdgeList( )
    { }
};

#ifndef NDEBUG
template <typename T>
inline void writeArray( const char* filename, const T* debug_ptr, int size )
{
    std::ofstream of( filename );

    for( int i=0; i<size; i++ ) {
        of << debug_ptr[i].debug_out( ) << std::endl;
    }
}

template <>
inline void writeArray<int2>( const char* filename, const int2* debug_ptr, int size )
{
    std::ofstream of( filename );

    for( int i=0; i<size; i++ ) {
        of << debug_ptr[i].x << " " << debug_ptr[i].y << std::endl;
    }
}
#endif // NDEBUG

template <typename T>
struct EdgeList
{
    DevEdgeList<T>  dev;
    HostEdgeList<T> host;

#ifndef NDEBUG
    void debugOut( int maxSize, const char* outFilename )
    {
        POP_CUDA_MEMCPY_TO_HOST_SYNC( &host.size, dev.size, sizeof(int) );

        const int size = min( maxSize, host.size );

        typedef T T_t;
        T* debug_ptr = new T_t [size];

        if( host.size > 0 ) {
            POP_CUDA_MEMCPY_SYNC( debug_ptr,
                                  dev.ptr,
                                  size * sizeof(T),
                                  cudaMemcpyDeviceToHost );
        }

        writeArray( outFilename, debug_ptr, size );
        delete [] debug_ptr;
    }
#endif // NDEBUG
};

}; // namespace popart


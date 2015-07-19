#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "triple_point.h"
#include "debug_macros.hpp"

namespace popart {

#ifndef NDEBUG
enum EdgeListFilter
{
    EdgeListFilterAny = 0,
    EdgeListFilterCommittedOnly = 1
};
#endif // NDEBUG

template <typename T>
struct DevEdgeList
{
    T*   ptr;
    int* size;

    __device__
    int Size() const
    {
        assert(size);
        return *size;
    }

    __device__
    void setSize( int sz )
    {
        assert(size);
        *size = sz;
    }
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
inline bool writeArrayFilter( const T& elem, EdgeListFilter f )
{
    std::cerr << "Applying filter to incompatible type in "
              << __FILE__ << ":" << __LINE__ << std::endl;
    return true;
}

template <>
inline bool writeArrayFilter<TriplePoint>( const TriplePoint& elem, EdgeListFilter f )
{
    switch( f ) {
    case EdgeListFilterCommittedOnly :
        return ( elem._coords_idx != 0 );
    default :
        std::cerr << "Applying unknown filter to type TriplePoint in "
                  << __FILE__ << ":" << __LINE__ << std::endl;
        break;
    }
    return true;
}

template <typename T>
inline void writeArray( std::vector<T>& out,
                        const T*        debug_ptr,
                        int             size,
                        EdgeListFilter  f = EdgeListFilterAny )
{
    if( f == EdgeListFilterAny ) {
        for( int i=0; i<size; i++ ) {
            out.push_back( debug_ptr[i] );
        }
    } else {
        for( int i=0; i<size; i++ ) {
            if( writeArrayFilter( debug_ptr[i], f ) ) {
                out.push_back( debug_ptr[i] );
            }
        }
    }
}

#if 0
template <typename T>
inline void writeArray( const char* filename, const T* debug_ptr, int size, const int* indexlist, int indexsize )
{
    std::ofstream of( filename );

    for( int idx=0; idx<indexsize; idx++ ) {
        if( idx < size ) {
            of << debug_ptr[idx].debug_out( ) << std::endl;
        } else {
            of << "index out of range" << std::endl;
        }
    }
}
#endif

template <typename T>
inline void writeArray( std::vector<T>& out, const T* debug_ptr, int debug_size, const int* indexlist, int indexsize )
{
    for( int i=0; i<indexsize; i++ ) {
        int idx = indexlist[i];
        if( idx < debug_size ) {
            out.push_back( debug_ptr[idx] );
        }
    }
}

#if 0
template <typename T>
inline void writeArray( const char* filename, const T* debug_ptr, int size,
                        EdgeListFilter  f = EdgeListFilterAny )
{
    std::ofstream of( filename );

    if( f == EdgeListFilterAny ) {
        for( int i=0; i<size; i++ ) {
            of << debug_ptr[i].debug_out( ) << std::endl;
        }
    } else {
        for( int i=0; i<size; i++ ) {
            if( writeArrayFilter( debug_ptr[i], f ) ) {
                of << debug_ptr[i].debug_out( ) << std::endl;
            }
        }
    }
}
#endif
#endif // NDEBUG

template <typename T>
struct EdgeList
{
    DevEdgeList<T>  dev;
    HostEdgeList<T> host;

#ifndef NDEBUG
    T* debug_ptr;

    EdgeList( ) : debug_ptr(0) { }
    ~EdgeList( ) { delete [] debug_ptr; }

private:
    bool get_debug_mem( int maxSize )
    {
        POP_CUDA_MEMCPY_TO_HOST_SYNC( &host.size, dev.size, sizeof(int) );

        if( host.size <= 0 ) return false;

        if( debug_ptr == 0 ) {
            const int size = min( maxSize, host.size );

            // An nvcc compiler bug requires typedef
            typedef T T_t;
            debug_ptr = new T_t [size];

            POP_CUDA_MEMCPY_SYNC( debug_ptr,
                                  dev.ptr,
                                  size * sizeof(T),
                                  cudaMemcpyDeviceToHost );
        }
        return true;
    }
public:
    const int* getDebugPtr( const int maxSize, int& size )
    {
        size = 0;
        bool success = get_debug_mem( maxSize );
        if( success ) {
            size = host.size;
            return debug_ptr;
        } else {
            return 0;
        }
    }

    void debug_out( int maxSize, std::vector<T>& out, EdgeListFilter f = EdgeListFilterAny )
    {
        bool success = get_debug_mem( maxSize );
        if( not success ) return;

        const int size = min( maxSize, host.size );
        writeArray( out, debug_ptr, size, f );
    }

    void debug_out( EdgeList<int>& indices, int maxSize, const char* outFilename )
    {
        int        indexsize = 0;
        const int* indexlist = indices.getDebugPtr( maxSize, indexsize );
        if( indexsize == 0 ) return;
        if( indexlist == 0 ) return;

        bool success = get_debug_mem( maxSize );
        if( not success ) return;

        const int size = min( maxSize, host.size );
        writeArray( outFilename, debug_ptr, size, indexlist, indexsize );
    }

    void debug_out( EdgeList<int>& indices, int maxSize, std::vector<T>& out )
    {
        int        indexsize = 0;
        const int* indexlist = indices.getDebugPtr( maxSize, indexsize );
        if( indexsize == 0 ) return;
        if( indexlist == 0 ) return;

        bool success = get_debug_mem( maxSize );
        if( not success ) return;

        const int size = min( maxSize, host.size );
        writeArray( out, debug_ptr, size, indexlist, indexsize );
    }
#endif // NDEBUG
};

}; // namespace popart


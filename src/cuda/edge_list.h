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

enum EdgeListAllocMode
{
    EdgeListDevOnly = 0,
    EdgeListBoth = 1
};

template <typename T> struct EdgeList;

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

protected:
    friend class EdgeList<T>;

    __host__
    void alloc( int sz )
    {
        void*  aptr;

        POP_CUDA_MALLOC( &aptr, sz*sizeof(T) );
        ptr = (T*)aptr;

        POP_CUDA_MALLOC( &aptr, sizeof(int) );
        size = (int*)aptr;
    }

    __host__
    void init( int sz, cudaStream_t stream )
    {
        POP_CUDA_MEMSET_ASYNC( ptr,
                               0,
                               sz * sizeof(T),
                               stream );
    }

    __host__
    void release( )
    {
        POP_CUDA_FREE( ptr );
        POP_CUDA_FREE( size );
        ptr = 0;
        size = 0;
    }
};

template <typename T>
struct HostEdgeList
{
    T*  ptr;
    int size;

    HostEdgeList()
        : ptr(0)
        , size(0)
    { }

    ~HostEdgeList( )
    {
        delete [] ptr;
    }

protected:
    friend class EdgeList<T>;

    __host__
    void alloc( int sz )
    {
        if( ptr == 0 ) {
            ptr = new T[sz];
        }
    }

    __host__
    void init( int sz )
    {
        if( ptr ) memset( ptr, 0, sz*sizeof(T) );
    }

    __host__
    void release( )
    {
        delete [] ptr;
        ptr = 0;
    }
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

#endif // NDEBUG

template <typename T>
struct EdgeList
{
private:
    int             alloc_num;
public:
    DevEdgeList<T>  dev;
    HostEdgeList<T> host;

    EdgeList( ) { }
    ~EdgeList( ) { }

    __host__
    void copySizeFromDevice( )
    {
        POP_CUDA_MEMCPY_TO_HOST_SYNC( &host.size, dev.size, sizeof(int) );
    }

    __host__
    void copySizeFromDevice( cudaStream_t stream )
    {
        POP_CUDA_MEMCPY_TO_HOST_ASYNC( &host.size, dev.size, sizeof(int), stream );
    }

    __host__
    void copyDataFromDevice( int sz )
    {
        POP_CUDA_MEMCPY_SYNC( host.ptr,
                              dev.ptr,
                              sz * sizeof(T),
                              cudaMemcpyDeviceToHost );
    }

    __host__
    void copyDataFromDevice( int sz, cudaStream_t stream )
    {
        POP_CUDA_MEMCPY_ASYNC( host.ptr,
                               dev.ptr,
                               sz * sizeof(T),
                               cudaMemcpyDeviceToHost,
                               stream );
    }

    __host__
    void alloc( int sz, EdgeListAllocMode mode )
    {
        alloc_num = sz;
        dev.alloc( sz );
        if( mode == EdgeListBoth ) host.alloc( sz );
    }

    __host__
    void init( cudaStream_t stream )
    {
        dev .init( alloc_num, stream );
        host.init( alloc_num );
    }

    __host__
    void release( )
    {
        dev .release();
        host.release();
    }
#ifndef NDEBUG
private:
    bool get_debug_mem( int maxSize )
    {
        copySizeFromDevice( );

        if( host.size <= 0 ) return false;

        if( host.ptr == 0 ) {
            const int size = min( maxSize, host.size );

            host.alloc( size );
            copyDataFromDevice( size );
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
            return host.ptr;
        } else {
            return 0;
        }
    }

public:
    void debug_out( int maxSize, std::vector<T>& out, EdgeListFilter f = EdgeListFilterAny )
    {
        bool success = get_debug_mem( maxSize );
        if( not success ) return;

        const int size = min( maxSize, host.size );
        writeArray( out, host.ptr, size, f );
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
        writeArray( outFilename, host.ptr, size, indexlist, indexsize );
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
        writeArray( out, host.ptr, size, indexlist, indexsize );
    }
#endif // NDEBUG
};

}; // namespace popart


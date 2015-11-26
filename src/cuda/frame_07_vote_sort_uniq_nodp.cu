#include "onoff.h"

// #include <iostream>
// #include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
// #include <stdio.h>
#include "debug_macros.hpp"
// #include "debug_is_on_edge.h"

#include "frame.h"
#include "assist.h"

using namespace std;

namespace popart
{

#ifdef USE_SEPARABLE_COMPILATION_IN_GRADDESC
// nothing
#else // USE_SEPARABLE_COMPILATION_IN_GRADDESC
__host__
bool Frame::applyVoteSortNoDP( const cctag::Parameters& params )
{
    cudaError_t err;

    POP_CUDA_MEMCPY_TO_HOST_ASYNC( &_vote._seed_indices.host.size,
                                   _vote._seed_indices.dev.getSizePtr(),
                                   sizeof(int), _stream );
    POP_CUDA_SYNC( _stream );

    if( _vote._seed_indices.host.size <= 0 ) return false;

    /* Note: we use the intermediate picture plane, _d_intermediate, as assist
     *       buffer for CUB algorithms. It is extremely likely that this plane
     *       is large enough in all cases. If there are any problems, call
     *       the function with assist_buffer=0, and the function will return
     *       the required size in assist_buffer_sz (call by reference).
     */
    void*  assist_buffer = (void*)_d_intermediate.data;
    size_t assist_buffer_sz;

#ifndef RADIX_WITHOUT_DOUBLEBUFFER
    // CUB IN CUDA 7.0 allowed only the DoubleBuffer interface
    cub::DoubleBuffer<int> d_keys( _vote._seed_indices.dev.ptr,
                                   _vote._seed_indices_2.dev.ptr );
#endif // not RADIX_WITHOUT_DOUBLEBUFFER

#ifdef CUB_INIT_CALLS
    assist_buffer_sz  = 0;

#ifdef RADIX_WITHOUT_DOUBLEBUFFER
    err = cub::DeviceRadixSort::SortKeys<int>( 0,
                                              assist_buffer_sz,
                                              _vote._seed_indices.dev.ptr,
                                              _vote._seed_indices_2.dev.ptr,
                                              _vote._seed_indices.host.size,
                                              0,             // begin_bit
                                              sizeof(int)*8, // end_bit
                                              _stream,
                                              DEBUG_CUB_FUNCTIONS );
#else // RADIX_WITHOUT_DOUBLEBUFFER
    err = cub::DeviceRadixSort::SortKeys<int>( 0,
                                              assist_buffer_sz,
                                              d_keys,
                                              _vote._seed_indices.host.size,
                                              0,             // begin_bit
                                              sizeof(int)*8, // end_bit
                                              _stream,
                                              DEBUG_CUB_FUNCTIONS );
#endif // RADIX_WITHOUT_DOUBLEBUFFER
	if( err != cudaSuccess ) {
	    std::cerr << "cub::DeviceRadixSort::SortKeys init step failed. Crashing." << std::endl;
	    std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
	    exit(-1);
	}
	if( assist_buffer_sz >= _d_intermediate.step * _d_intermediate.rows ) {
        std::cerr << "cub::DeviceRadixSort::SortKeys requires too much intermediate memory. Crashing." << std::endl;
	    exit( -1 );
	}
#else // not CUB_INIT_CALLS
    assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;
#endif // not CUB_INIT_CALLS

#ifdef RADIX_WITHOUT_DOUBLEBUFFER
    err = cub::DeviceRadixSort::SortKeys( assist_buffer,
                                          assist_buffer_sz,
                                          _vote._seed_indices.dev.ptr,
                                          _vote._seed_indices_2.dev.ptr,
                                          _vote._seed_indices.host.size,
                                          0,             // begin_bit
                                          sizeof(int)*8, // end_bit
                                          _stream,
                                          DEBUG_CUB_FUNCTIONS );
    POP_CUDA_SYNC( _stream );
    POP_CUDA_FATAL_TEST( err, "CUB SortKeys failed" );

    std::swap( _vote._seed_indices.dev.ptr,           _vote._seed_indices_2.dev.ptr );
    std::swap( _vote._seed_indices.dev.getSizePtr(),  _vote._seed_indices_2.dev.getSizePtr() );
#else // RADIX_WITHOUT_DOUBLEBUFFER
    err = cub::DeviceRadixSort::SortKeys( assist_buffer,
                                          assist_buffer_sz,
                                          d_keys,
                                          _vote._seed_indices.host.size,
                                          0,             // begin_bit
                                          sizeof(int)*8, // end_bit
                                          _stream,
                                          DEBUG_CUB_FUNCTIONS );
    POP_CUDA_SYNC( _stream );
    POP_CUDA_FATAL_TEST( err, "CUB SortKeys failed" );

    /* After SortKeys, both buffers in d_keys have been altered.
     * The final result is stored in d_keys.d_buffers[d_keys.selector].
     * The other buffer is invalid.
     */
    if( d_keys.d_buffers[d_keys.selector] == _vote._seed_indices_2.dev.ptr ) {
        std::swap( _vote._seed_indices.dev.ptr,   _vote._seed_indices_2.dev.ptr );
        std::swap( _vote._seed_indices.dev.getSizePtr(),  _vote._seed_indices_2.dev.getSizePtr() );
    }
#endif // RADIX_WITHOUT_DOUBLEBUFFER
    return true;
}

__host__
void Frame::applyVoteUniqNoDP( const cctag::Parameters& params )
{
    cudaError_t err;

    void*  assist_buffer = (void*)_d_intermediate.data;
    size_t assist_buffer_sz;

#ifdef CUB_INIT_CALLS
	assist_buffer_sz  = 0;
	// std::cerr << "before cub::DeviceSelect::Unique(0)" << std::endl;
    err = cub::DeviceSelect::Unique<int*,int*,int*>(
        0,
        assist_buffer_sz,
        _vote._seed_indices.dev.ptr,     // input
        _vote._seed_indices_2.dev.ptr,   // output
        _vote._seed_indices_2.dev.getSizePtr(),  // output
        _vote._seed_indices.host.size,   // input (unchanged in sort)
        _stream,
        DEBUG_CUB_FUNCTIONS );

	if( err != cudaSuccess ) {
	    std::cerr << "cub::DeviceSelect::Unique init step failed. Crashing." << std::endl;
	    std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
	    exit(-1);
	}
	if( assist_buffer_sz >= _d_intermediate.step * _d_intermediate.rows ) {
            std::cerr << "cub::DeviceSelect::Unique requires too much intermediate memory. Crashing." << std::endl;
	    exit( -1 );
	}
#else // not CUB_INIT_CALLS
    assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;
#endif // not CUB_INIT_CALLS

    /* Unique ensure that we check every "chosen" point only once.
     * Output is in _vote._seed_indices_2.dev
     */
    err = cub::DeviceSelect::Unique<int*,int*,int*>(
        assist_buffer,
        assist_buffer_sz,
        _vote._seed_indices.dev.ptr,     // input
        _vote._seed_indices_2.dev.ptr,   // output
        _vote._seed_indices_2.dev.getSizePtr(),  // output
        _vote._seed_indices.host.size,   // input (unchanged in sort)
        _stream,
        DEBUG_CUB_FUNCTIONS );

    POP_CHK_CALL_IFSYNC;
    POP_CUDA_SYNC( _stream );
    POP_CUDA_FATAL_TEST( err, "CUB Unique failed" );
}
#endif // USE_SEPARABLE_COMPILATION_IN_GRADDESC

} // namespace popart


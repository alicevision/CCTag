/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "onoff.h"

#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "debug_macros.hpp"

#include "frame.h"
#include "framemeta.h"
#include "assist.h"

using namespace std;

namespace cctag
{

#ifdef USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ
// nothing
#else // USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ
__host__
bool Frame::applyVoteSortNoDP( )
{
    cudaError_t err;

    _inner_points.copySizeFromDevice( _stream, EdgeListWait );

    if( _inner_points.host.size <= 0 ) {
        return false;
    }

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
    cub::DoubleBuffer<int> d_keys( _inner_points.dev.ptr,
                                   _interm_inner_points.dev.ptr );
#endif // not RADIX_WITHOUT_DOUBLEBUFFER

    assist_buffer_sz  = 0;

#ifdef RADIX_WITHOUT_DOUBLEBUFFER
    err = cub::DeviceRadixSort::SortKeys<int>( 0,
                                              assist_buffer_sz,
                                              _inner_points.dev.ptr,
                                              _interm_inner_points.dev.ptr,
                                              _inner_points.host.size,
                                              0,             // begin_bit
                                              sizeof(int)*8, // end_bit
                                              _stream,
                                              DEBUG_CUB_FUNCTIONS );
#else // RADIX_WITHOUT_DOUBLEBUFFER
    err = cub::DeviceRadixSort::SortKeys<int>( 0,
                                              assist_buffer_sz,
                                              d_keys,
                                              _inner_points.host.size,
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

#ifdef RADIX_WITHOUT_DOUBLEBUFFER
    err = cub::DeviceRadixSort::SortKeys( assist_buffer,
                                          assist_buffer_sz,
                                          _inner_points.dev.ptr,
                                          _interm_inner_points.dev.ptr,
                                          _inner_points.host.size,
                                          0,             // begin_bit
                                          sizeof(int)*8, // end_bit
                                          _stream,
                                          DEBUG_CUB_FUNCTIONS );
    POP_CUDA_SYNC( _stream );
    POP_CUDA_FATAL_TEST( err, "CUB SortKeys failed" );

    std::swap( _inner_points.dev.ptr,           _interm_inner_points.dev.ptr );
#else // RADIX_WITHOUT_DOUBLEBUFFER
    err = cub::DeviceRadixSort::SortKeys( assist_buffer,
                                          assist_buffer_sz,
                                          d_keys,
                                          _inner_points.host.size,
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
    if( d_keys.d_buffers[d_keys.selector] == _interm_inner_points.dev.ptr ) {
        /* data pointers must be swapped, we need the buffer later */
        std::swap( _inner_points.dev.ptr,   _interm_inner_points.dev.ptr );

        /* The sizes are irrelevant: they cannot change in sorting */
    }
#endif // RADIX_WITHOUT_DOUBLEBUFFER
    return true;
}

__host__
void Frame::applyVoteUniqNoDP( )
{
    cudaError_t err;

    void*  assist_buffer = (void*)_d_intermediate.data;
    size_t assist_buffer_sz;

	assist_buffer_sz  = 0;
	// std::cerr << "before cub::DeviceSelect::Unique(0)" << std::endl;

    err = cub::DeviceSelect::Unique<int*,int*,int*>(
        0,
        assist_buffer_sz,
        _inner_points.dev.ptr,        // input
        _interm_inner_points.dev.ptr, // output
        _d_interm_int,                // output
        _inner_points.host.size,      // input (unchanged in sort)
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

    /* Unique ensure that we check every "chosen" point only once.
     * Output is in _interm_inner_points.dev
     */
    err = cub::DeviceSelect::Unique<int*,int*,int*>(
        assist_buffer,
        assist_buffer_sz,
        _inner_points.dev.ptr,        // input
        _interm_inner_points.dev.ptr, // output
        _d_interm_int,                // output
        _inner_points.host.size,      // input (unchanged in sort)
        _stream,
        DEBUG_CUB_FUNCTIONS );

    _meta.toDevice_D2S( List_size_interm_inner_points, _d_interm_int, _stream );

    POP_CHK_CALL_IFSYNC;
    POP_CUDA_SYNC( _stream );
    POP_CUDA_FATAL_TEST( err, "CUB Unique failed: " );
}

bool Frame::applyVoteSortUniq( )
{
    bool success = applyVoteSortNoDP( );
    if( not success ) return false;
    applyVoteUniqNoDP( );
    return true;
}

#endif // USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ

} // namespace cctag


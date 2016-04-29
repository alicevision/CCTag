#include "onoff.h"

#ifdef USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ

#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "assist.h"

using namespace std;

namespace popart
{

namespace descent
{

__global__
void dp_call_03_sort_uniq(
    FrameMetaPtr             meta, // input
    DevEdgeList<int>         inner_points,   // input/output
    DevEdgeList<int>         interm_inner_points,  // invalidated buffer
    cv::cuda::PtrStepSzb     intermediate ) // invalidated buffer
{
    cudaError_t  err;

    int listsize = meta.list_size_inner_points();

    if( listsize == 0 ) return;

    assert( meta.list_size_inner_points() > 0 );

    /* Note: we use the intermediate picture plane, _d_intermediate, as assist
     *       buffer for CUB algorithms. It is extremely likely that this plane
     *       is large enough in all cases. If there are any problems, call
     *       the function with assist_buffer=0, and the function will return
     *       the required size in assist_buffer_sz (call by reference).
     */
    void*  assist_buffer = (void*)intermediate.data;
    size_t assist_buffer_sz = intermediate.step * intermediate.rows;

    cub::DoubleBuffer<int> keys( inner_points.ptr,
                                 interm_inner_points.ptr );

    /* After SortKeys, both buffers in d_keys have been altered.
     * The final result is stored in d_keys.d_buffers[d_keys.selector].
     * The other buffer is invalid.
     */
#ifdef CUB_INIT_CALLS
    assist_buffer_sz  = 0;

    err = cub::DeviceRadixSort::SortKeys( 0,
                                          assist_buffer_sz,
                                          keys,
                                          listsize,
                                          0,             // begin_bit
                                          sizeof(int)*8, // end_bit
                                          0,   // use stream 0
                                          DEBUG_CUB_FUNCTIONS );
    if( err != cudaSuccess ) {
        meta.list_size_interm_inner_points() = 0;
        return;
    }
    if( assist_buffer_sz > intermediate.step * intermediate.rows ) {
        meta.list_size_interm_inner_points() = 0;
        return;
    }
#else // not CUB_INIT_CALLS
    assist_buffer_sz = intermediate.step * intermediate.rows;
#endif // not CUB_INIT_CALLS

    err = cub::DeviceRadixSort::SortKeys( assist_buffer,
                                          assist_buffer_sz,
                                          keys,
                                          listsize,
                                          0,             // begin_bit
                                          sizeof(int)*8, // end_bit
                                          0,   // use stream 0
                                          DEBUG_CUB_FUNCTIONS );        // synchronous for debugging

    cudaDeviceSynchronize( );
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        return;
    }

    assert( meta.list_size_inner_points() > 0 );

    if( keys.Current() == interm_inner_points.ptr ) {
        int* swap_ptr    = interm_inner_points.ptr;
        interm_inner_points.ptr = inner_points.ptr;
        inner_points.ptr  = swap_ptr;

        meta.swap_buffers_after_sort() = 1;
    } else {
        meta.swap_buffers_after_sort() = 0;
    }

    meta.list_size_interm_inner_points() = listsize;

    assert( interm_inner_points.ptr != 0 );
    assert( inner_points.ptr != 0 );

#ifdef CUB_INIT_CALLS
    assist_buffer_sz = 0;

    err = cub::DeviceSelect::Unique( 0,
                                     assist_buffer_sz,
                                     inner_points.ptr,     // input
                                     interm_inner_points.ptr,   // output
                                     &meta.list_size_interm_inner_points(), // output
                                     meta.list_size_inner_points(), // input (unchanged in sort)
                                     0,  // use stream 0
                                     DEBUG_CUB_FUNCTIONS ); // synchronous for debugging
    if( err != cudaSuccess ) {
        meta.list_size_interm_inner_points() = 0;
        return;
    }
    if( assist_buffer_sz > intermediate.step * intermediate.rows ) {
        meta.list_size_interm_inner_points() = 0;
        return;
    }
#else // not CUB_INIT_CALLS
    // safety: SortKeys is allowed to alter assist_buffer_sz
    assist_buffer_sz = intermediate.step * intermediate.rows;
#endif // not CUB_INIT_CALLS

    /* Unique ensure that we check every "chosen" point only once.
     * Output is in _interm_inner_points.dev
     */
    err = cub::DeviceSelect::Unique( assist_buffer,
                                     assist_buffer_sz,
                                     inner_points.ptr,     // input
                                     interm_inner_points.ptr,   // output
                                     &meta.list_size_interm_inner_points(), // output
                                     meta.list_size_inner_points(), // input (unchanged in sort)
                                     0,  // use stream 0
                                     DEBUG_CUB_FUNCTIONS ); // synchronous for debugging

}

} // namespace descent

__host__
bool Frame::applyVoteSortUniq( )
{
    descent::dp_call_03_sort_uniq
        <<<1,1,0,_stream>>>
        ( _meta,
          _inner_points.dev,
          _interm_inner_points.dev,
          cv::cuda::PtrStepSzb(_d_intermediate) ); // buffer
    POP_CHK_CALL_IFSYNC;

    int mustSwap;

    _meta.fromDevice( Swap_buffers_after_sort, mustSwap, _stream );
    POP_CHK_CALL_IFSYNC;

    cudaError_t err = cudaStreamSynchronize( _stream );
    POP_CUDA_FATAL_TEST( err, "Failed to synchronize after sort: " );
    if( mustSwap ) {
        std::swap( _inner_points.dev.ptr, _interm_inner_points.dev.ptr );
    }

#ifndef NDEBUG
    _interm_inner_points.copySizeFromDevice( _stream, EdgeListCont );
    _inner_points.copySizeFromDevice( _stream, EdgeListWait );
    cudaDeviceSynchronize();
    std::cerr << __func__ << " l " << _layer << ": DP"
         << " # pts before srt/uniq: " << _inner_points.host.size
         << " # pts after srt/uniq: " << _interm_inner_points.host.size << endl;
#endif
    return true;
}

} // namespace popart

#else // not USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ
// other file
#endif // not USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ


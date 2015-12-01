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
    cudaStream_t childStream;
    cudaStreamCreateWithFlags( &childStream, cudaStreamNonBlocking );

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
                                          childStream,   // use stream 0
                                          DEBUG_CUB_FUNCTIONS );
    if( err != cudaSuccess ) {
        meta.list_size_interm_inner_points() = 0;
        cudaStreamDestroy( childStream );
        return;
    }
    if( assist_buffer_sz > intermediate.step * intermediate.rows ) {
        meta.list_size_interm_inner_points() = 0;
        cudaStreamDestroy( childStream );
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
                                          childStream,   // use stream 0
                                          DEBUG_CUB_FUNCTIONS );        // synchronous for debugging

    cudaDeviceSynchronize( );
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        cudaStreamDestroy( childStream );
        return;
    }

    assert( meta.list_size_inner_points() > 0 );

    if( keys.Current() == interm_inner_points.ptr ) {
        int* swap_ptr    = interm_inner_points.ptr;
        interm_inner_points.ptr = inner_points.ptr;
        inner_points.ptr  = swap_ptr;
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
                                     childStream,  // use stream 0
                                     DEBUG_CUB_FUNCTIONS ); // synchronous for debugging
    if( err != cudaSuccess ) {
        meta.list_size_interm_inner_points() = 0;
        cudaStreamDestroy( childStream );
        return;
    }
    if( assist_buffer_sz > intermediate.step * intermediate.rows ) {
        meta.list_size_interm_inner_points() = 0;
        cudaStreamDestroy( childStream );
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
                                     childStream,  // use stream 0
                                     DEBUG_CUB_FUNCTIONS ); // synchronous for debugging

    cudaStreamDestroy( childStream );
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
    return true;
}

} // namespace popart

#else // not USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ
// other file
#endif // not USE_SEPARABLE_COMPILATION_FOR_SORT_UNIQ


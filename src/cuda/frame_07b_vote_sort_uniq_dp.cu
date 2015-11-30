#include "onoff.h"

#ifdef USE_SEPARABLE_COMPILATION

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
                DevEdgeList<int>         seedIndices,   // input/output
                DevEdgeList<int>         seedIndices2,  // invalidated buffer
                cv::cuda::PtrStepSzb     intermediate ) // invalidated buffer
{
    cudaError_t  err;
    cudaStream_t childStream;
    cudaStreamCreateWithFlags( &childStream, cudaStreamNonBlocking );

    int listsize = seedIndices.getSize();

    if( listsize == 0 ) return;

    assert( seedIndices.getSize() > 0 );

    /* Note: we use the intermediate picture plane, _d_intermediate, as assist
     *       buffer for CUB algorithms. It is extremely likely that this plane
     *       is large enough in all cases. If there are any problems, call
     *       the function with assist_buffer=0, and the function will return
     *       the required size in assist_buffer_sz (call by reference).
     */
    void*  assist_buffer = (void*)intermediate.data;
    size_t assist_buffer_sz = intermediate.step * intermediate.rows;

    cub::DoubleBuffer<int> keys( seedIndices.ptr,
                                 seedIndices2.ptr );

    /* After SortKeys, both buffers in d_keys have been altered.
     * The final result is stored in d_keys.d_buffers[d_keys.selector].
     * The other buffer is invalid.
     */
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
        return;
    }

    assert( seedIndices.getSize() > 0 );

    if( keys.Current() == seedIndices2.ptr ) {
        int* swap_ptr    = seedIndices2.ptr;
        seedIndices2.ptr = seedIndices.ptr;
        seedIndices.ptr  = swap_ptr;
    }

    seedIndices2.setSize( listsize );

    assert( seedIndices2.ptr != 0 );
    assert( seedIndices.ptr != 0 );
    assert( seedIndices.getSize() > 0 );

    // safety: SortKeys is allowed to alter assist_buffer_sz
    assist_buffer_sz = intermediate.step * intermediate.rows;

    /* Unique ensure that we check every "chosen" point only once.
     * Output is in _interm_inner_points.dev
     */
    err = cub::DeviceSelect::Unique( assist_buffer,
                                     assist_buffer_sz,
                                     seedIndices.ptr,     // input
                                     seedIndices2.ptr,   // output
                                     seedIndices2.getSizePtr(),  // output
                                     seedIndices.getSize(), // input (unchanged in sort)
                                     childStream,  // use stream 0
                                     DEBUG_CUB_FUNCTIONS ); // synchronous for debugging

    cudaStreamDestroy( childStream );
}

} // namespace descent

__host__
bool Frame::applyVoteSortUniqDP( const cctag::Parameters& params )
{
    descent::dp_call_03_sort_uniq
        <<<1,1,0,_stream>>>
        (
          _inner_points.dev,        // output
          _interm_inner_points.dev,      // buffer
          cv::cuda::PtrStepSzb(_d_intermediate) ); // buffer
    POP_CHK_CALL_IFSYNC;
    return true;
}

} // namespace popart

#else // not USE_SEPARABLE_COMPILATION
// other file
#endif // not USE_SEPARABLE_COMPILATION


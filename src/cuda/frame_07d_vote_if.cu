#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "frameparam.h"
#include "assist.h"
#include "onoff.h"

using namespace std;

namespace popart {

struct NumVotersIsGreaterEqual
{
    DevEdgeList<TriplePoint> _array;
    int                      _compare;

    __host__ __device__
    __forceinline__
    NumVotersIsGreaterEqual( DevEdgeList<TriplePoint> _d_array )
        : _array( _d_array )
    {}

    __device__
    __forceinline__
    bool operator()(const int &a) const {
        return (_array.ptr[a]._winnerSize >= tagParam.minVotesToSelectCandidate );
    }
};

#ifdef USE_SEPARABLE_COMPILATION_FOR_VOTE_IF

__global__
void dp_call_vote_if(
    DevEdgeList<TriplePoint> voters,        // input
    DevEdgeList<int>         seedIndices,   // output
    DevEdgeList<int>         seedIndices2,  // input
    cv::cuda::PtrStepSzb     intermediate ) // buffer
{
    /* Filter all chosen inner points that have fewer
     * voters than required by Parameters.
     */

    cudaError_t err;

    if( seedIndices.getSize() == 0 ) {
        seedIndices2.setSize(0);
        return;
    }

    NumVotersIsGreaterEqual select_op( voters );

#ifdef CUB_INIT_CALLS
    size_t assist_buffer_sz = 0;
    err = cub::DeviceSelect::If( 0,
                                 assist_buffer_sz,
                                 seedIndices2.ptr,
                                 seedIndices.ptr,
                                 seedIndices.getSizePtr(),
                                 seedIndices2.getSize(),
                                 select_op,
                                 0,     // use stream 0
                                 DEBUG_CUB_FUNCTIONS ); // synchronous for debugging
    if( err != cudaSuccess ) {
        return;
    }
    if( assist_buffer_sz > intermediate.step * intermediate.rows ) {
        seedIndices.setSize(0);
        return;
    }
#else // not CUB_INIT_CALLS
    size_t assist_buffer_sz = intermediate.step * intermediate.rows;
#endif // not CUB_INIT_CALLS
    void*  assist_buffer = (void*)intermediate.data;

    cub::DeviceSelect::If( assist_buffer,
                           assist_buffer_sz,
                           seedIndices2.ptr,
                           seedIndices.ptr,
                           seedIndices.getSizePtr(),
                           seedIndices2.getSize(),
                           select_op,
                           0,     // use stream 0
                           DEBUG_CUB_FUNCTIONS ); // synchronous for debugging
}

__host__
bool Frame::applyVoteIf( )
{
    dp_call_vote_if
        <<<1,1,0,_stream>>>
        ( _voters.dev,  // input
          _vote._seed_indices.dev,        // output
          _vote._seed_indices_2.dev,      // input
          cv::cuda::PtrStepSzb(_d_intermediate) ); // buffer
    POP_CHK_CALL_IFSYNC;

    _vote._seed_indices.copySizeFromDevice( _stream, EdgeListCont );

    return true;
}
#else // not USE_SEPARABLE_COMPILATION_FOR_VOTE_IF
__host__
bool Frame::applyVoteIf( )
{
    cudaError_t err;

    void*  assist_buffer = (void*)_d_intermediate.data;
    size_t assist_buffer_sz;

    NumVotersIsGreaterEqual select_op( _voters.dev );
#ifdef CUB_INIT_CALLS
    assist_buffer_sz  = 0;
    err = cub::DeviceSelect::If( 0,
                                 assist_buffer_sz,
                                 _vote._seed_indices_2.dev.ptr,
                                 _vote._seed_indices.dev.ptr,
                                 _vote._seed_indices.dev.getSizePtr(),
                                 _vote._seed_indices_2.host.size,
                                 select_op,
                                 _stream,
                                 DEBUG_CUB_FUNCTIONS );

    POP_CUDA_FATAL_TEST( err, "CUB DeviceSelect::If failed in init test" );

    if( assist_buffer_sz >= _d_intermediate.step * _d_intermediate.rows ) {
        std::cerr << "cub::DeviceSelect::If requires too much intermediate memory. Crashing." << std::endl;
        exit( -1 );
    }
#else
    // THIS CODE WORKED BEFORE
    assist_buffer_sz = _d_intermediate.step * _d_intermediate.rows;
#endif

    /* Filter all chosen inner points that have fewer
     * voters than required by Parameters.
     */
    err = cub::DeviceSelect::If( assist_buffer,
                                 assist_buffer_sz,
                                 _vote._seed_indices_2.dev.ptr,
                                 _vote._seed_indices.dev.ptr,
                                 _vote._seed_indices.dev.getSizePtr(),
                                 _vote._seed_indices_2.host.size,
                                 select_op,
                                 _stream,
                                 DEBUG_CUB_FUNCTIONS );
    POP_CHK_CALL_IFSYNC;
    POP_CUDA_FATAL_TEST( err, "CUB DeviceSelect::If failed" );

    _vote._seed_indices.copySizeFromDevice( _stream, EdgeListCont );
    return true;
}
#endif // not USE_SEPARABLE_COMPILATION_FOR_VOTE_IF

} // namespace popart


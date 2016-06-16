#include <iostream>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "frameparam.h"
#include "assist.h"
#include "onoff.h"

using namespace std;

namespace popart {

class NumVotersIsGreaterEqual
{
    DevEdgeList<CudaEdgePoint> _edgepoints;

public:
    __host__ __device__
    __forceinline__
    NumVotersIsGreaterEqual( DevEdgeList<CudaEdgePoint> a )
        : _edgepoints( a )
    {}

    __device__
    __forceinline__
    bool operator()(const int &a) const {
        return (_edgepoints.ptr[a]._dev_winnerSize >= tagParam.minVotesToSelectCandidate );
    }
};

#ifdef USE_SEPARABLE_COMPILATION_FOR_VOTE_IF

__global__
void dp_call_vote_if(
    FrameMetaPtr               meta,
    DevEdgeList<CudaEdgePoint> all_edgecoords,        // input
    DevEdgeList<int>           voters,        // input
    DevEdgeList<int>           inner_points,   // output
    DevEdgeList<int>           interm_inner_points,  // input
    cv::cuda::PtrStepSzb       intermediate ) // buffer
{
    /* Filter all chosen inner points that have fewer
     * voters than required by Parameters.
     */

    cudaError_t err;

    if( meta.list_size_interm_inner_points() == 0 ) {
        meta.list_size_inner_points() = 0;
        return;
    }

    NumVotersIsGreaterEqual select_op( all_edgecoords );

#ifdef CUB_INIT_CALLS
    size_t assist_buffer_sz = 0;
    err = cub::DeviceSelect::If( 0,
                                 assist_buffer_sz,
                                 interm_inner_points.ptr,
                                 inner_points.ptr,
                                 &meta.list_size_inner_points(),
                                 meta.list_size_interm_inner_points(),
                                 select_op,
                                 0,     // use stream 0
                                 DEBUG_CUB_FUNCTIONS ); // synchronous for debugging
    if( err != cudaSuccess ) {
        return;
    }
    if( assist_buffer_sz > intermediate.step * intermediate.rows ) {
        meta.list_size_inner_points() = 0;
        return;
    }
#else // not CUB_INIT_CALLS
    size_t assist_buffer_sz = intermediate.step * intermediate.rows;
#endif // not CUB_INIT_CALLS
    void*  assist_buffer = (void*)intermediate.data;

    cub::DeviceSelect::If( assist_buffer,
                           assist_buffer_sz,
                           interm_inner_points.ptr,
                           inner_points.ptr,
                           &meta.list_size_inner_points(),
                           meta.list_size_interm_inner_points(),
                           select_op,
                           0,     // use stream 0
                           DEBUG_CUB_FUNCTIONS ); // synchronous for debugging
}

__host__
bool Frame::applyVoteIf( )
{
    dp_call_vote_if
        <<<1,1,0,_stream>>>
        ( _meta,
          _edgepoints.dev, // input
          _voters.dev,  // input
          _inner_points.dev,        // output
          _interm_inner_points.dev,      // input
          cv::cuda::PtrStepSzb(_d_intermediate) ); // buffer
    POP_CHK_CALL_IFSYNC;

    _inner_points.copySizeFromDevice( _stream, EdgeListCont );

    return true;
}
#else // not USE_SEPARABLE_COMPILATION_FOR_VOTE_IF
__host__
bool Frame::applyVoteIf( )
{
    if( _interm_inner_points.host.size == 0 ) {
        return false;
    }

    cudaError_t err;

    void*  assist_buffer = (void*)_d_intermediate.data;
    size_t assist_buffer_sz;

    NumVotersIsGreaterEqual select_op( _edgepoints.dev );
#ifdef CUB_INIT_CALLS
    assist_buffer_sz  = 0;
    err = cub::DeviceSelect::If( 0,
                                 assist_buffer_sz,
                                 _interm_inner_points.dev.ptr,
                                 _inner_points.dev.ptr,
                                 _d_interm_int,
                                 _interm_inner_points.host.size,
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
                                 _interm_inner_points.dev.ptr,
                                 _inner_points.dev.ptr,
                                 _d_interm_int,
                                 _interm_inner_points.host.size,
                                 select_op,
                                 _stream,
                                 DEBUG_CUB_FUNCTIONS );
    POP_CHK_CALL_IFSYNC;
    POP_CUDA_FATAL_TEST( err, "CUB DeviceSelect::If failed" );

    _meta.toDevice_D2S( List_size_inner_points, _d_interm_int, _stream );
    _inner_points.copySizeFromDevice( _stream, EdgeListCont );
    return true;
}
#endif // not USE_SEPARABLE_COMPILATION_FOR_VOTE_IF

} // namespace popart


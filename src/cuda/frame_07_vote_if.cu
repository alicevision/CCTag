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
#include "onoff.h"

using namespace std;

namespace popart {
namespace vote {

#ifdef USE_SEPARABLE_COMPILATION
__global__
void dp_call_05_if( DevEdgeList<int2>        edgeCoords, // input
                cv::cuda::PtrStepSzb     edgeImage, // input
                cv::cuda::PtrStepSz16s   dx, // input
                cv::cuda::PtrStepSz16s   dy, // input
                DevEdgeList<TriplePoint> chainedEdgeCoords, // output
                cv::cuda::PtrStepSz32s   edgepointIndexTable, // output
                DevEdgeList<int>         seedIndices, // output
                DevEdgeList<int>         seedIndices2, // output
                cv::cuda::PtrStepSzb     intermediate, // buffer
                const uint32_t           param_nmax, // input param
                const int32_t            param_thrGradient, // input param
                const size_t             param_numCrowns, // input param
                const float              param_ratioVoting, // input param
                const int                param_minVotesToSelectCandidate ) // input param
{
    if( seedIndices.getSize() == 0 ) {
        seedIndices2.setSize(0);
        return;
    }

    cudaStream_t childStream;
    cudaStreamCreateWithFlags( &childStream, cudaStreamNonBlocking );

    // safety: SortKeys is allowed to alter assist_buffer_sz
    void*  assist_buffer = (void*)intermediate.data;
    size_t assist_buffer_sz = intermediate.step * intermediate.rows;

    /* Filter all chosen inner points that have fewer
     * voters than required by Parameters.
     */
    NumVotersIsGreaterEqual select_op( param_minVotesToSelectCandidate,
                                       chainedEdgeCoords );
    cub::DeviceSelect::If( assist_buffer,
                           assist_buffer_sz,
                           seedIndices2.ptr,
                           seedIndices.ptr,
                           seedIndices.getSizePtr(),
                           seedIndices2.getSize(),
                           select_op,
                           childStream,     // use stream 0
                           DEBUG_CUB_FUNCTIONS ); // synchronous for debugging

    cudaStreamDestroy( childStream );
}

} // namespace vote
#endif // USE_SEPARABLE_COMPILATION

#ifdef USE_SEPARABLE_COMPILATION
__host__
bool Frame::applyVoteIf( const cctag::Parameters& params )
{
    descent::dp_call_05_if
        <<<1,1,0,_stream>>>
        ( _vote._all_edgecoords.dev,      // input
          _d_edges,                       // input
          _d_dx,                          // input
          _d_dy,                          // input
          _vote._chained_edgecoords.dev,  // output
          _vote._d_edgepoint_index_table, // output
          _vote._seed_indices.dev,        // output
          _vote._seed_indices_2.dev,      // buffer
          cv::cuda::PtrStepSzb(_d_intermediate), // buffer
          params._distSearch,             // input param
          params._thrGradientMagInVote,   // input param
          params._nCrowns,                // input param
          params._ratioVoting,            // input param
          params._minVotesToSelectCandidate ); // input param
    POP_CHK_CALL_IFSYNC;
    return true;
}
#else // not USE_SEPARABLE_COMPILATION
__host__
void Frame::applyVoteIf( const cctag::Parameters& params )
{
    void*  assist_buffer = (void*)_d_intermediate.data;
    size_t assist_buffer_sz;

    NumVotersIsGreaterEqual select_op( params._minVotesToSelectCandidate,
                                       _vote._chained_edgecoords.dev );
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

    if( err != cudaSuccess ) {
        std::cerr << "cub::DeviceSelect::If init step failed. Crashing." << std::endl;
        std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
        exit(-1);
    }
    if( assist_buffer_sz >= _d_intermediate.step * _d_intermediate.rows ) {
        std::cerr << "cub::DeviceSelect::If requires too much intermediate memory. Crashing." << std::endl;
        exit( -1 );
    }
#else
    // THIS CODE WORKED BEFORE
    // safety: SortKeys is allowed to alter assist_buffer_sz
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

    _vote._seed_indices.copySizeFromDevice( _stream );
}
#endif // not USE_SEPARABLE_COMPILATION

} // namespace popart


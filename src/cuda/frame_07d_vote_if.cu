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

#include <boost/thread/mutex.hpp>

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

__global__
void extractVotes( int numInnerPoints, const int* innerPointIndex, int64_t* numVotesOut, DevEdgeList<TriplePoint> voters, int rows, int cols )
{
    const int idx = blockIdx.x * 32 + threadIdx.x;

    if( idx >= numInnerPoints ) {
        return;
    }

    const int offset = innerPointIndex[idx];

    /* we are sorting Descending.
     * dominant: number of votes that a point got
     * second:   small Y coord
     * third:    small X coord
     */
    const int votes  = voters.ptr[offset]._winnerSize * cols * rows
                     + ( rows - voters.ptr[offset].coord.y ) * cols
                     + ( cols - voters.ptr[offset].coord.x );

    numVotesOut[idx] = votes;
}

#ifdef USE_SEPARABLE_COMPILATION_FOR_VOTE_IF

__global__
void dp_call_vote_if(
    FrameMetaPtr             meta,
    DevEdgeList<TriplePoint> voters,        // input
    DevEdgeList<int>         inner_points,   // output
    DevEdgeList<int>         interm_inner_points,  // input
    cv::cuda::PtrStepSzb     intermediate ) // buffer
{
    /* Filter all chosen inner points that have fewer
     * voters than required by Parameters.
     */

    cudaError_t err;

    if( meta.list_size_interm_inner_points() == 0 ) {
        meta.list_size_inner_points() = 0;
        return;
    }

    NumVotersIsGreaterEqual select_op( voters );

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
          _voters.dev,  // input
          _inner_points.dev,        // output
          _interm_inner_points.dev,      // input
          cv::cuda::PtrStepSzb(_d_intermediate) ); // buffer
    POP_CHK_CALL_IFSYNC;

    _inner_points.copySizeFromDevice( _stream, EdgeListWait );

    if( _inner_points.host.size == 0 )
    {
        return true;
    }

    /* Create an index array that contains the number of voters for each
     * point in _inner_points (_inner_points contains indices into the
     * array voters).
     */
    int64_t* vote_collection_1 = (int64_t*)_d_intermediate.ptr(0);
    int64_t* vote_collection_2 = (int64_t*)_d_intermediate.ptr(_d_intermediate.rows/2);

    if( _d_intermediate.rows/2 * _d_intermediate.step < _inner_points.host.size * sizeof(int64_t) ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    _d_intermediate is too small for sorting seed candidates" << std::endl;
        exit( -1 );
    }

    dim3 block( 32, 1, 1 );
    dim3 grid( grid_divide( _inner_points.host.size, 32 ), 1, 1 );

    extractVotes
        <<<grid,block>>>
        ( _inner_points.host.size,
          _inner_points.dev.ptr,
          vote_collection_1,
          _voters.dev,
          _d_intermediate.rows,
          _d_intermediate.cols );

    POP_CHK;

    void*  assist_buffer = (void*)_d_map.data;
    size_t assist_buffer_sz = 0;

    cub::DoubleBuffer<int64_t> keys( vote_collection_1,
                                     vote_collection_2 );
    cub::DoubleBuffer<int> values( _inner_points.dev.ptr,
                                   _interm_inner_points.dev.ptr );


    cudaError_t err;
    err = cub::DeviceRadixSort::SortPairsDescending(
                                           0,
                                           assist_buffer_sz,
                                           keys,
                                           values,
                                           _inner_points.host.size,
                                           0,
                                           sizeof(int64_t)*8,
                                           0, // default stream
                                           DEBUG_CUB_FUNCTIONS );

    if( err != cudaSuccess ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs init step failed. Crashing." << std::endl;
        std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
        exit(-1);
    }
    if( assist_buffer_sz >= _d_map.step * _d_map.rows ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs requires too much intermediate memory. Crashing." << std::endl;
        exit( -1 );
    }

    err = cub::DeviceRadixSort::SortPairsDescending(
                                           assist_buffer,
                                           assist_buffer_sz,
                                           keys,
                                           values,
                                           _inner_points.host.size,
                                           0,
                                           sizeof(int64_t)*8,
                                           0, // default stream
                                           DEBUG_CUB_FUNCTIONS );

    if( err != cudaSuccess ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs compute step failed. Crashing." << std::endl;
        std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
        exit(-1);
    }

    if( values.Current() != _inner_points.dev.ptr ) {
        std::swap( _inner_points.dev.ptr, _interm_inner_points.dev.ptr );
    }

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

    NumVotersIsGreaterEqual select_op( _voters.dev );

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
    _inner_points.copySizeFromDevice( _stream, EdgeListWait );

    if( _inner_points.host.size == 0 )
    {
        return true;
    }

    /* Create an index array that contains the number of voters for each
     * point in _inner_points (_inner_points contains indices into the
     * array voters).
     */
    int64_t* vote_collection_1 = (int64_t*)_d_intermediate.ptr(0);
    int64_t* vote_collection_2 = (int64_t*)_d_intermediate.ptr(_d_intermediate.rows/2);

    if( _d_intermediate.rows/2 * _d_intermediate.step < _inner_points.host.size * sizeof(int64_t) ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    _d_intermediate is too small for sorting seed candidates" << std::endl;
        exit( -1 );
    }

    dim3 block( 32, 1, 1 );
    dim3 grid( grid_divide( _inner_points.host.size, 32 ), 1, 1 );

    extractVotes
        <<<grid,block>>>
        ( _inner_points.host.size,
          _inner_points.dev.ptr,
          vote_collection_1,
          _voters.dev,
          _d_intermediate.rows,
          _d_intermediate.cols );

    POP_CHK;

    assist_buffer = (void*)_d_map.data;
    assist_buffer_sz = 0;

    cub::DoubleBuffer<int64_t> keys( vote_collection_1,
                               vote_collection_2 );
    cub::DoubleBuffer<int> values( _inner_points.dev.ptr,
                                   _interm_inner_points.dev.ptr );

    err = cub::DeviceRadixSort::SortPairsDescending( 0,
                                           assist_buffer_sz,
                                           keys,
                                           values,
                                           _inner_points.host.size,
                                           0,
                                           sizeof(int64_t)*8,
                                           0, // default stream
                                           DEBUG_CUB_FUNCTIONS );

    if( err != cudaSuccess ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs init step failed. Crashing." << std::endl;
        std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
        exit(-1);
    }
    if( assist_buffer_sz >= _d_map.step * _d_map.rows ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs requires too much intermediate memory. Crashing." << std::endl;
        exit( -1 );
    }

    err = cub::DeviceRadixSort::SortPairsDescending( assist_buffer,
                                           assist_buffer_sz,
                                           keys,
                                           values,
                                           _inner_points.host.size,
                                           0,
                                           sizeof(int64_t)*8,
                                           0, // default stream
                                           DEBUG_CUB_FUNCTIONS );

    if( err != cudaSuccess ) {
        std::cerr << "cub::DeviceRadixSort::SortPairs compute step failed. Crashing." << std::endl;
        std::cerr << "Error message: " << cudaGetErrorString( err ) << std::endl;
        exit(-1);
    }

    if( values.Current() != _inner_points.dev.ptr ) {
        std::swap( _inner_points.dev.ptr, _interm_inner_points.dev.ptr );
    }

    return true;
}

#endif // not USE_SEPARABLE_COMPILATION_FOR_VOTE_IF

} // namespace popart


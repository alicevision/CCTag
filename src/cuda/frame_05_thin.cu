#include <cuda_runtime.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "assist.h"

namespace popart
{

using namespace std;

namespace thinning {

static unsigned char h_lut[256] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 
        1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 
};

// Note that the transposed h_lut_t is not really necessary
// because flipping the 4 LSBs and 4 HSBs in the unsigned char that
// I use for lookup is fast.
static unsigned char h_lut_t[256] = {
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 
        1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 
        1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
};

__device__ __constant__ unsigned char d_lut[256];

__device__ __constant__ unsigned char d_lut_t[256];

__device__
bool update_pixel( const int idx, const int idy, cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst, bool first_run )
{
    unsigned char result = 0;
    if( src.ptr(idy)[idx] == 2 &&
        idx >= 1 && idy >=1 && idx <= src.cols-2 && idy <= src.rows-2 ) {
        uint8_t log = 0;

        log |= ( src.ptr(idy-1)[idx  ] == 2 ) ? 0x01 : 0;
        log |= ( src.ptr(idy-1)[idx+1] == 2 ) ? 0x02 : 0;
        log |= ( src.ptr(idy  )[idx+1] == 2 ) ? 0x04 : 0;
        log |= ( src.ptr(idy+1)[idx+1] == 2 ) ? 0x08 : 0;
        log |= ( src.ptr(idy+1)[idx  ] == 2 ) ? 0x10 : 0;
        log |= ( src.ptr(idy+1)[idx-1] == 2 ) ? 0x20 : 0;
        log |= ( src.ptr(idy  )[idx-1] == 2 ) ? 0x40 : 0;
        log |= ( src.ptr(idy-1)[idx-1] == 2 ) ? 0x80 : 0;

        if( first_run ) {
            result = d_lut[log] ? 2 : 0;
        } else {
            result = d_lut_t[log];
        }
    }
    __syncthreads();
    dst.ptr(idy)[idx] = result;
    return ( result != 0 );
}

__global__
void first_round( cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst )
{
    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    update_pixel( idx, idy, src, dst, true );
}

__global__
void second_round( cv::cuda::PtrStepSzb src,          // input
                   cv::cuda::PtrStepSzb dst,          // output
#ifndef NDEBUG
                   DevEdgeList<int2>    edgeCoords,   // output
                   FrameMetaPtr         meta )
#else
                   DevEdgeList<int2>    edgeCoords )  // output
#endif
{
    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    bool keep = update_pixel( idx, idy, src, dst, false );

    if( keep ) {
        atomicAdd( &meta.num_edges_thinned(), 1 );
    }
#if 0
    uint32_t write_index;
    if( keep ) {
        write_index = atomicAdd( edgeCoords.getSizePtr(), 1 );
    }
#else
    uint32_t mask = __ballot( keep );  // bitfield of warps with results
    uint32_t ct   = __popc( mask );    // horizontal reduce
    uint32_t leader = __ffs(mask) - 1; // the highest thread id with indicator==true
    uint32_t write_index;
    if( threadIdx.x == leader ) {
        // leader gets warp's offset from global value and increases it
        write_index = atomicAdd( edgeCoords.getSizePtr(), int(ct) );
    }
    write_index = __shfl( write_index, leader ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index
#endif

    if( keep ) {
        if( write_index < EDGE_POINT_MAX ) {
            edgeCoords.ptr[write_index] = make_int2( idx, idy );
        }
    }
}

__global__
void set_null( DevEdgeList<int2> edgeCoords )
{
    edgeCoords.setSize( 0 );
}

__global__
void set_edgemax( DevEdgeList<int2> edgeCoords )
{
    if( edgeCoords.Size() > EDGE_POINT_MAX ) {
        edgeCoords.setSize( EDGE_POINT_MAX );
    }
}

}; // namespace thinning

__host__
void Frame::initThinningTable( )
{
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( thinning::d_lut,
                                         thinning::h_lut,
                                         256*sizeof(unsigned char) );
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( thinning::d_lut_t,
                                         thinning::h_lut_t,
                                         256*sizeof(unsigned char) );
}

__host__
void Frame::applyThinning( )
{
    dim3 block( 32, 1, 1 );
    dim3 grid( grid_divide( getWidth(), 32 ),
               getHeight(),
               1 );

    thinning::first_round
        <<<grid,block,0,_stream>>>
        ( _d_hyst_edges, cv::cuda::PtrStepSzb(_d_intermediate) );
    POP_CHK_CALL_IFSYNC;

    thinning::set_null
        <<<1,1,0,_stream>>>
        ( _vote._all_edgecoords.dev );

    // POP_CUDA_SET0_ASYNC( _vote._all_edgecoords.dev.getSizePtr(), _stream );

#ifndef NDEBUG
    _meta.toDevice( Num_edges_thinned, 0, _stream );

    thinning::second_round
        <<<grid,block,0,_stream>>>
        ( cv::cuda::PtrStepSzb(_d_intermediate), // input
          _d_edges,                              // output
          _vote._all_edgecoords.dev,             // output
          _meta );

    int val;
    _meta.fromDevice( Num_edges_thinned, val, _stream );
    _vote._all_edgecoords.copySizeFromDevice( _stream, EdgeListWait );
    std::cerr << __FILE__ << ":" << __LINE__ << std::endl
              << "num of edge points after thinning: " << val << std::endl
              << "num of edge points added to list:  " << _vote._all_edgecoords.host.size << std::endl
              << "edgemax: " << EDGE_POINT_MAX << std::endl;
    _vote._all_edgecoords.copyDataFromDeviceSync( );

#else // NDEBUG
    thinning::second_round
        <<<grid,block,0,_stream>>>
        ( cv::cuda::PtrStepSzb(_d_intermediate), // input
          _d_edges,                              // output
          _vote._all_edgecoords.dev );           // output
#endif // NDEBUG

    thinning::set_edgemax
        <<<1,1,0,_stream>>>
        ( _vote._all_edgecoords.dev );

    _vote._all_edgecoords.copySizeFromDevice( _stream, EdgeListCont );
#if 0
    debugPointIsOnEdge( _d_edges, _vote._all_edgecoords, _stream );
#endif // NDEBUG

#ifdef EDGE_LINKING_HOST_SIDE
    /* After thinning_and_store, _all_edgecoords is no longer changed.
     * Make a non-blocking copy the number of items in the list to the host.
     */
    cudaEventRecord( _download_ready_event.edgecoords1, _stream );
#endif // EDGE_LINKING_HOST_SIDE
}

__host__
void Frame::applyThinDownload( )
{
#ifdef EDGE_LINKING_HOST_SIDE
    /* After thinning_and_store, _all_edgecoords is no longer changed
     * we can copy it to the host for edge linking
     */
    // cudaStreamWaitEvent( _download_stream, _download_ready_event.edgecoords1, 0 );

    /* CPU must wait for counter _vote._all_edgecoords.host.size */
    cudaEventSynchronize( _download_ready_event.edgecoords1 );
    POP_CHK_CALL_IFSYNC;

    // cudaEventSynchronize( _download_ready_event.edgecoords2 );
    _vote._all_edgecoords.copyDataFromDeviceAsync( _download_stream );
    POP_CHK_CALL_IFSYNC;
#ifndef NDEBUG
    if( _vote._all_edgecoords.host.size <= 0 ) {
        // initialize the hostside array to 0 for debugging
        _vote._all_edgecoords.initHost( );
    }
#endif // NDEBUG
#endif // EDGE_LINKING_HOST_SIDE
}

}; // namespace popart


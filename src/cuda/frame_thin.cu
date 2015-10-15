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
    if( src.ptr(idy)[idx] != 2 ) {
        dst.ptr(idy)[idx] = 0;
        return false;
    }

    if( idx >= 1 && idy >=1 && idx <= src.cols-2 && idy <= src.rows-2 ) {
        uint8_t log = 0;

        log |= ( src.ptr(idy-1)[idx  ] == 2 ) ? 0x01 : 0;
        log |= ( src.ptr(idy-1)[idx+1] == 2 ) ? 0x02 : 0;
        log |= ( src.ptr(idy  )[idx+1] == 2 ) ? 0x04 : 0;
        log |= ( src.ptr(idy+1)[idx+1] == 2 ) ? 0x08 : 0;
        log |= ( src.ptr(idy+1)[idx  ] == 2 ) ? 0x10 : 0;
        log |= ( src.ptr(idy+1)[idx-1] == 2 ) ? 0x20 : 0;
        log |= ( src.ptr(idy  )[idx-1] == 2 ) ? 0x40 : 0;
        log |= ( src.ptr(idy-1)[idx-1] == 2 ) ? 0x80 : 0;

        unsigned char result;
        if( first_run ) {
            result = d_lut[log] ? 2 : 0;
        } else {
            result = d_lut_t[log];
        }
        dst.ptr(idy)[idx] = result;

        return ( result != 0 );
        // return true;
    }
    return false;
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
                   DevEdgeList<int2>    edgeCoords,   // output
                   uint32_t             param_edgeMax )     // input
{
    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    bool keep = update_pixel( idx, idy, src, dst, false );

    uint32_t mask = __ballot( keep );  // bitfield of warps with results
    uint32_t ct   = __popc( mask );    // horizontal reduce
    uint32_t leader = __ffs(mask) - 1; // the highest thread id with indicator==true
    uint32_t write_index;
    if( threadIdx.x == leader ) {
        // leader gets warp's offset from global value and increases it
        write_index = atomicAdd( edgeCoords.size, int(ct) );
    }
    write_index = __shfl( write_index, leader ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index

    if( keep && write_index < param_edgeMax ) {
        edgeCoords.ptr[write_index] = make_int2( idx, idy );
    }
}

__global__
void set_edgemax( DevEdgeList<int2> edgeCoords,
                  uint32_t          param_edgeMax )
{
    if( edgeCoords.Size() > param_edgeMax ) {
        edgeCoords.setSize( param_edgeMax );
    }
}

#ifdef USE_SEPARABLE_COMPILATION
__global__
void dp_caller( const size_t         width,          // input
                const size_t         height,         // input
                cv::cuda::PtrStepSzb hystEdges,      // input
                cv::cuda::PtrStepSzb edges,          // output
                DevEdgeList<int2>    edgeCoords,     // output
                cv::cuda::PtrStepSzb intermediate,   // intermediate
                uint32_t             param_edgeMax ) // input param
{
    edgeCoords.size = 0;

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = ( width / 32 ) + ( width % 32 == 0 ? 0 : 1 );
    grid.y  = height;

    first_round
        <<<grid,block>>>
        ( hystEdges, intermediate );

    second_round
        <<<grid,block>>>
        ( intermediate,    // input
          edges,           // output
          edgeCoords,      // output
          param_edgeMax ); // input param
}
#endif // USE_SEPARABLE_COMPILATION

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
void Frame::applyThinning( const cctag::Parameters & params )
{
#ifdef USE_SEPARABLE_COMPILATION
    thinning::dp_caller
        <<<1,1,0,_stream>>>
        ( getWidth(),
          getHeight(),
          _d_hyst_edges,                         // input
          _d_edges,                              // output
          _vote._all_edgecoords.dev,             // output
          cv::cuda::PtrStepSzb(_d_intermediate), // intermediate
          params._maxEdges );                    // input param
    POP_CHK_CALL_IFSYNC;
#else // USE_SEPARABLE_COMPILATION
    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = ( getWidth() / 32 ) + ( getWidth() % 32 == 0 ? 0 : 1 );
    grid.y  = getHeight();

    thinning::first_round
        <<<grid,block,0,_stream>>>
        ( _d_hyst_edges, cv::cuda::PtrStepSzb(_d_intermediate) );
    POP_CHK_CALL_IFSYNC;

    POP_CUDA_SET0_ASYNC( _vote._all_edgecoords.dev.size, _stream );

    thinning::secound_round
        <<<grid,block,0,_stream>>>
        ( cv::cuda::PtrStepSzb(_d_intermediate), // input
          _d_edges,                              // output
          _vote._all_edgecoords.dev,             // output
          params._maxEdges );                    // input
    POP_CHK_CALL_IFSYNC;
#endif // USE_SEPARABLE_COMPILATION

    thinning::set_edgemax
        <<<1,1,0,_stream>>>
        ( _vote._all_edgecoords.dev,
          params._maxEdges );
    POP_CHK_CALL_IFSYNC;

#ifndef NDEBUG
    debugPointIsOnEdge( _d_edges, _vote._all_edgecoords, _stream );
#endif // NDEBUG

#ifdef EDGE_LINKING_HOST_SIDE
    /* After thinning_and_store, _all_edgecoords is no longer changed.
     * Make a non-blocking copy the number of items in the list to the host.
     */
    _vote._all_edgecoords.copySizeFromDevice( _stream );
    cudaEventRecord( &_download_ready_event.edgecoords, _stream );
    POP_CHK_CALL_IFSYNC;
#endif // EDGE_LINKING_HOST_SIDE
}

__host__
void Frame::applyThinDownload( const cctag::Parameters& )
{
#ifdef EDGE_LINKING_HOST_SIDE
    /* After thinning_and_store, _all_edgecoords is no longer changed
     * we can copy it to the host for edge linking
     */

    /* CPU must wait for counter _vote._all_edgecoords.host.size */
    cudaEventSynchronize( _download_ready_event.edgecoords );
    POP_CHK_CALL_IFSYNC;
    if( _vote._all_edgecoords.host.size > 0 ) {
        _vote._all_edgecoords.copyDataFromDevice( _vote._all_edgecoords.host.size,
                                                  _download_stream );
        POP_CHK_CALL_IFSYNC;
    }
#ifndef NDEBUG
    else
    {
        // initialize the hostside array to 0 for debugging
        _vote._all_edgecoords.initHost( );
    }
#endif // NDEBUG
#endif // EDGE_LINKING_HOST_SIDE
}

}; // namespace popart


#include <cuda_runtime.h>
#include "debug_macros.hpp"

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
void second_round( FrameMetaPtr               meta,
                   cv::cuda::PtrStepSzb       src,              // input
                   cv::cuda::PtrStepSz16s     d_dx,             // input
                   cv::cuda::PtrStepSz16s     d_dy,             // input
                   cv::cuda::PtrStepSzb       dst,              // output
                   DevEdgeList<CudaEdgePoint> d_edgepoints,     // output
                   cv::cuda::PtrStepSz32s     d_edgepoint_map ) // output
{
    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    bool keep = update_pixel( idx, idy, src, dst, false );

    uint32_t mask = __ballot( keep );  // bitfield of warps with results
    uint32_t ct   = __popc( mask );    // horizontal reduce
    uint32_t leader = __ffs(mask) - 1; // the highest thread id with indicator==true
    int      write_index;
    if( threadIdx.x == leader ) {
        // leader gets warp's offset from global value and increases it
        write_index = atomicAdd( &meta.list_size_edgepoints(), int(ct) );
    }
    write_index = __shfl( write_index, leader ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index

    bool predicate = ( keep && ( write_index < EDGE_POINT_MAX ) );

    d_edgepoint_map.ptr(idy)[idx] = predicate ? write_index : -1;

    if( predicate ) {
        const short dx = d_dx.ptr(idy)[idx];
        const short dy = d_dy.ptr(idy)[idx];
        d_edgepoints.ptr[write_index].init( idx, idy, dx, dy );
    }
}

__global__
void set_edgemax( FrameMetaPtr meta )
{
    if( meta.list_size_edgepoints() > EDGE_POINT_MAX ) {
        meta.list_size_edgepoints() = EDGE_POINT_MAX;
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

    _meta.toDevice( List_size_edgepoints, 0, _stream );

    thinning::second_round
        <<<grid,block,0,_stream>>>
        ( _meta,
          cv::cuda::PtrStepSzb(_d_intermediate), // input
          _d_dx,                                 // input
          _d_dy,                                 // input
          _d_edges,                              // output
          _edgepoints.dev,                       // output
          _d_edgepoint_map );                    // output

    thinning::set_edgemax
        <<<1,1,0,_stream>>>
        ( _meta );

    _edgepoints.copySizeFromDevice( _stream, EdgeListCont );

    /* After thinning_and_store, _edgepoints is no longer changed.
     * Make a non-blocking copy the number of items in the list to the host.
     */
    cudaEventRecord( _download_ready_event.edgecoords1, _stream );
}

__host__
void Frame::applyThinDownload( )
{
    /* After thinning_and_store, _edgepoints is no longer changed
     * we can copy it to the host for edge linking
     */
    // cudaStreamWaitEvent( _download_stream, _download_ready_event.edgecoords1, 0 );

    /* CPU must wait for counter _edgepoints.host.size */
    cudaEventSynchronize( _download_ready_event.edgecoords1 );
    POP_CHK_CALL_IFSYNC;

    // cudaEventSynchronize( _download_ready_event.edgecoords2 );
    _edgepoints.copyDataFromDeviceAsync( _download_stream );
    POP_CHK_CALL_IFSYNC;
#ifndef NDEBUG
    if( _edgepoints.host.size <= 0 ) {
        // initialize the hostside array to 0 for debugging
        _edgepoints.initHost( );
    }
#endif // NDEBUG
}

}; // namespace popart


#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"
#include "assist.h"

namespace popart
{

using namespace std;

static unsigned char h_thinning_lut[256] = {
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

// Note that the transposed h_thinning_lut_t is not really necessary
// because flipping the 4 LSBs and 4 HSBs in the unsigned char that
// I use for lookup is really quick. Therefore: remove soon.
static unsigned char h_thinning_lut_t[256] = {
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

__device__ __constant__ unsigned char d_thinning_lut[256];

__device__ __constant__ unsigned char d_thinning_lut_t[256];

__global__
void compute_mag_l1( cv::cuda::PtrStepSz16s src_dx,
                     cv::cuda::PtrStepSz16s src_dy,
                     cv::cuda::PtrStepSz32u dst )
{
    int block_x = blockIdx.x * 32;
    int idx     = block_x + threadIdx.x;
    int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, dst ) ) return;

    int16_t dx = src_dx.ptr(idy)[idx];
    int16_t dy = src_dy.ptr(idy)[idx];
    dx = d_abs( dx );
    dy = d_abs( dy );
    dst.ptr(idy)[idx] = dx + dy;
}

__global__
void compute_mag_l2( cv::cuda::PtrStepSz16s src_dx,
                     cv::cuda::PtrStepSz16s src_dy,
                     cv::cuda::PtrStepSz32u dst )
{
    int block_x = blockIdx.x * 32;
    int idx     = block_x + threadIdx.x;
    int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, dst ) ) return;

    int16_t dx = src_dx.ptr(idy)[idx];
    int16_t dy = src_dy.ptr(idy)[idx];
    // --- hypot --
    dx *= dx;
    dy *= dy;
    dst.ptr(idy)[idx] = __fsqrt_rz( (float)( dx + dy ) );
}

__global__
void compute_map( const cv::cuda::PtrStepSz16s dx,
                  const cv::cuda::PtrStepSz16s dy,
                  const cv::cuda::PtrStepSz32u mag,
                  cv::cuda::PtrStepSzb         map,
                  const float                  low_thresh,
                  const float                  high_thresh )
{
    const int CANNY_SHIFT = 15;
    const int TG22 = (int32_t)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, dx ) ) return;

    int32_t  dxVal  = dx.ptr(idy)[idx];
    int32_t  dyVal  = dy.ptr(idy)[idx];
    uint32_t magVal = mag.ptr(idy)[idx];

    // -1 if only is negative, 1 else
    // const int32_t signVal = (dxVal ^ dyVal) < 0 ? -1 : 1;
    const int32_t signVal = d_sign( dxVal ^ dyVal );

    dxVal = d_abs( dxVal );
    dyVal = d_abs( dyVal );

    // 0 - the pixel can not belong to an edge
    // 1 - the pixel might belong to an edge
    // 2 - the pixel does belong to an edge
    uint8_t edge_type = 0;

    if( magVal > low_thresh )
    {
        const int32_t tg22x = dxVal * TG22;
        const int32_t tg67x = tg22x + ((dxVal + dxVal) << CANNY_SHIFT);

        dyVal <<= CANNY_SHIFT;

        int2 x = (dyVal < tg22x) ? make_int2( idx - 1, idx + 1 )
                                 : (dyVal > tg67x ) ? make_int2( idx, idx )
                                                    : make_int2( idx - signVal, idx + signVal );
        int2 y = (dyVal < tg22x) ? make_int2( idy, idy )
                                 : make_int2( idy - 1, idy + 1 );

        x.x = clamp( x.x, dx.cols );
        x.y = clamp( x.y, dx.cols );
        y.x = clamp( y.x, dx.rows );
        y.y = clamp( y.y, dx.rows );

        if( magVal > mag.ptr(y.x)[x.x] && magVal >= mag.ptr(y.y)[x.y] ) {
            edge_type = 1 + (uint8_t)(magVal > high_thresh);
        }
    }
    __syncthreads();

    map.ptr(idy)[idx] = edge_type;
}

__device__
bool thinning_inner( const int idx, const int idy, cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst, bool first_run )
{
    if( src.ptr(idy)[idx] == 0 ) {
        dst.ptr(idy)[idx] = 0;
        return false;
    }

    if( idx >= 1 && idy >=1 && idx <= src.cols-2 && idy <= src.rows-2 ) {
        uint8_t log = 0;

        log |= ( src.ptr(idy-1)[idx  ] != 0 ) ? 0x01 : 0;
        log |= ( src.ptr(idy-1)[idx+1] != 0 ) ? 0x02 : 0;
        log |= ( src.ptr(idy  )[idx+1] != 0 ) ? 0x04 : 0;
        log |= ( src.ptr(idy+1)[idx+1] != 0 ) ? 0x08 : 0;
        log |= ( src.ptr(idy+1)[idx  ] != 0 ) ? 0x10 : 0;
        log |= ( src.ptr(idy+1)[idx-1] != 0 ) ? 0x20 : 0;
        log |= ( src.ptr(idy  )[idx-1] != 0 ) ? 0x40 : 0;
        log |= ( src.ptr(idy-1)[idx-1] != 0 ) ? 0x80 : 0;

#if 1
        if( first_run )
            dst.ptr(idy)[idx] = d_thinning_lut[log];
        else
            dst.ptr(idy)[idx] = d_thinning_lut_t[log];
#else
        if( first_run == false ) {
            uint8_t b = log;
            b   = ( b   << 4 ) & 0xf0;
            log = ( ( log >> 4 ) & 0x0f ) | b;
        }

        dst.ptr(idy)[idx] = d_thinning_lut[log];
#endif
        return true;
    }
    return false;
}

__global__
void thinning( cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst )
{
    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    thinning_inner( idx, idy, src, dst, true );
}

__global__
void thinning_and_store( cv::cuda::PtrStepSzb src,          // input
                         cv::cuda::PtrStepSzb dst,          // output
                         DevEdgeList<int2>    edgeCoords,   // output
                         uint32_t             edgeMax )     // input
{
    const int block_x = blockIdx.x * 32;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    bool keep = thinning_inner( idx, idy, src, dst, false );

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

    if( keep && write_index < edgeMax ) {
        edgeCoords.ptr[write_index] = make_int2( idx, idy );
    }
}

__host__
void Frame::initThinningTable( )
{
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_thinning_lut,
                                         h_thinning_lut,
                                         256*sizeof(unsigned char) );
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_thinning_lut_t,
                                         h_thinning_lut_t,
                                         256*sizeof(unsigned char) );
}

__host__
void Frame::applyMag( const cctag::Parameters & params )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = ( getWidth() / 32 ) + ( getWidth() % 32 == 0 ? 0 : 1 );
    grid.y  = getHeight();

    dim3 big_block;
    dim3 big_grid;
    big_block.x = 32;
    big_block.y = 32;
    big_grid.x  = ( getWidth()  / 32 ) + ( getWidth()  % 32 == 0 ? 0 : 1 );
    big_grid.y  = ( getHeight() / 32 ) + ( getHeight() % 32 == 0 ? 0 : 1 );

    // necessary to merge into 1 stream
    compute_mag_l2
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag );
    POP_CHK_CALL_IFSYNC;

    compute_map
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag, _d_map, 256.0f * params._cannyThrLow, 256.0f * params._cannyThrHigh );
    POP_CHK_CALL_IFSYNC;

    cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void Frame::applyThinning( const cctag::Parameters & params )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = ( getWidth() / 32 ) + ( getWidth() % 32 == 0 ? 0 : 1 );
    grid.y  = getHeight();

    thinning
        <<<grid,block,0,_stream>>>
        ( _d_hyst_edges, cv::cuda::PtrStepSzb(_d_intermediate) );
    POP_CHK_CALL_IFSYNC;

    POP_CUDA_SET0_ASYNC( _vote._all_edgecoords.dev.size, _stream );

    thinning_and_store
        <<<grid,block,0,_stream>>>
        ( cv::cuda::PtrStepSzb(_d_intermediate), // input
          _d_edges,                              // output
          _vote._all_edgecoords.dev,           // output
          params._maxEdges );                    // input
    POP_CHK_CALL_IFSYNC;

    cerr << "Leave " << __FUNCTION__ << endl;
}

}; // namespace popart


#include <iostream>
#include <limits>
#include <cuda_runtime.h>
#include <stdio.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"

namespace popart
{

using namespace std;

/*
 * This is actually a code file, to be included into frame.cu
 */

/* These numbers are taken from Lilian's file cctag/fiter/cvRecode.cpp
 * Note that the array looks like because a __constant__ device array
 * with 2 dimensions is conceptually very problematic. The reason is
 * that the compiler pads each dimension separately, but there is no
 * way of asking about this padding (pitch, stepsize, whatever you
 * call it).
 * If the kernels should be multi-use, we need one array with two offsets.
 * Aligning to anything less than 16 floats is a bad idea.
 */

static const float h_gauss_filter[32] =
{
    0.000053390535453f,
    0.001768051711852f,
    0.021539279301849f,
    0.096532352630054f,
    0.159154943091895f,
    0.096532352630054f,
    0.021539279301849f,
    0.001768051711852f,
    0.000053390535453f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.002683701023220f,
    0.066653979229454f,
    0.541341132946452f,
    1.213061319425269f,
    0.0f,
    -1.213061319425269f,
    -0.541341132946452f,
    -0.066653979229454f,
    -0.002683701023220f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f
};

#define GAUSS_TABLE  0 // Gauss parameters
#define GAUSS_DERIV 16 // first derivative
__device__ __constant__ float d_gauss_filter[32];

__device__ __constant__ float d_gauss_filter_by_256[16];

#define V7_WIDTH    32

template <class DestType>
__global__
void filter_gauss_horiz( cv::cuda::PtrStepSzf          src,
                         cv::cuda::PtrStepSz<DestType> dst,
                         int                           filter )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;
    int idx;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[filter + offset];

        idx = clamp( block_x + threadIdx.x - offset - 4, src.cols );
        float val = src.ptr(block_y)[idx];
        out += ( val * g );
    }

    if( block_y >= dst.rows ) return;
    if( idx*sizeof(float) >= dst.step ) return;

    bool nix = ( block_x + threadIdx.x >= dst.cols ) || ( block_y >= dst.rows );
    dst.ptr(block_y)[idx] = nix ? 0 : (DestType)out;
}

template <class DestType>
__global__
void filter_gauss_vert( cv::cuda::PtrStepSzf          src,
                        cv::cuda::PtrStepSz<DestType> dst,
                        int                           filter )
{
    const int block_x = blockIdx.x * V7_WIDTH;
    const int block_y = blockIdx.y;
    const int idx     = block_x + threadIdx.x;
    int idy;

    if( idx*sizeof(float) >= src.step ) return;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[filter + offset];

        idy = clamp( block_y - offset - 4, src.rows );
        float val = src.ptr(idy)[idx];
        out += ( val * g );
    }

    if( idy >= dst.rows ) return;

    bool nix = ( idx >= dst.cols );
    dst.ptr(idy)[idx] = nix ? 0 : (DestType)out;
}

__global__
void filter_gauss_horiz_from_uchar( cv::cuda::PtrStepSzb src,
                                    cv::cuda::PtrStepSzf dst )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;
    int idx;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        // float g  = d_gauss_filter_by_256[offset];
        float g  = d_gauss_filter[offset];

        idx = clamp( block_x + threadIdx.x - offset - 4, src.cols );
        float val = src.ptr(block_y)[idx];
        out += ( val * g );
    }

    if( block_y >= dst.rows ) return;
    if( idx * sizeof(float) >= dst.step ) return;

    bool nix = ( block_x + threadIdx.x >= dst.cols ) || ( block_y >= dst.rows );
    dst.ptr(block_y)[idx] = nix ? 0 : out;
}

__global__
void compute_mag_l1( cv::cuda::PtrStepSz16s src_dx,
                     cv::cuda::PtrStepSz16s src_dy,
                     cv::cuda::PtrStepSz32u dst )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int idx     = block_x + threadIdx.x;
    int idy     = blockIdx.y;

    if( idx >= dst.cols ) return;
    if( idy >= dst.rows ) return;

    int16_t dx = src_dx.ptr(idy)[idx];
    int16_t dy = src_dy.ptr(idy)[idx];
    dx = (dx < 0 ) ? -dx : dx;
    dy = (dy < 0 ) ? -dy : dy;
    dst.ptr(idy)[idx] = dx + dy;
}

__global__
void compute_mag_l2( cv::cuda::PtrStepSz16s src_dx,
                     cv::cuda::PtrStepSz16s src_dy,
                     cv::cuda::PtrStepSz32u dst )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int idx     = block_x + threadIdx.x;
    int idy     = blockIdx.y;

    if( idx >= dst.cols ) return;
    if( idy >= dst.rows ) return;

    int16_t dx = src_dx.ptr(idy)[idx];
    int16_t dy = src_dy.ptr(idy)[idx];
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

    const int block_x = blockIdx.x * V7_WIDTH;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    if( idx >= dx.cols ) return;
    if( idy >= dx.rows ) return;

    int32_t  dxVal  = dx.ptr(idy)[idx];
    int32_t  dyVal  = dy.ptr(idy)[idx];
    uint32_t magVal = mag.ptr(idy)[idx];

    // -1 if only is negative, 1 else
    const int32_t signVal = (dxVal ^ dyVal) < 0 ? -1 : 1;

    dxVal = (dxVal < 0 ) ? -dxVal : dxVal;
    dyVal = (dyVal < 0 ) ? -dyVal : dyVal;

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

__global__
void edge_hysteresis( cv::cuda::PtrStepSzb map, cv::cuda::PtrStepSzb edges )
{
    const int block_x = blockIdx.x * V7_WIDTH;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    uint8_t log = 0;

    if( idx >= 1 && idy >=1 && idx <= map.cols-2 && idy <= map.rows-2 ) {
        log |= ( map.ptr(idy-1)[idx-1] == 2 ) ? 0x80 : 0;
        log |= ( map.ptr(idy-1)[idx  ] == 2 ) ? 0x01 : 0;
        log |= ( map.ptr(idy-1)[idx+1] == 2 ) ? 0x02 : 0;
        log |= ( map.ptr(idy  )[idx+1] == 2 ) ? 0x04 : 0;
        log |= ( map.ptr(idy+1)[idx+1] == 2 ) ? 0x08 : 0;
        log |= ( map.ptr(idy+1)[idx  ] == 2 ) ? 0x10 : 0;
        log |= ( map.ptr(idy+1)[idx-1] == 2 ) ? 0x20 : 0;
        log |= ( map.ptr(idy  )[idx-1] == 2 ) ? 0x40 : 0;
        if( log != 0 ) log = 1;
    }
    edges.ptr(idy)[idx] = log;
}

#if 0
__global__
void debug_gauss( cv::cuda::PtrStepSzf src )
{
    size_t non_null_ct = 0;
    float minval = 1000.0f;
    float maxval = -1000.0f;
    for( size_t i=0; i<src.rows; i++ )
        for( size_t j=0; j<src.cols; j++ ) {
            float f = src.ptr(i)[j];
            if( f != 0.0f )
                non_null_ct++;
            minval = min( minval, f );
            maxval = max( maxval, f );
        }
    printf("There are %d non-null values in the Gaussian end result (min %f, max %f)\n", non_null_ct, minval, maxval );
}
#endif

__host__
void Frame::initGaussTable( )
{
    float h_gauss_filter_by_256[9];
    for( int i=0; i<9; i++ ) {
        h_gauss_filter_by_256[i] = h_gauss_filter[i] / 256.0f;
    }

    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_gauss_filter,
                                         h_gauss_filter,
                                         32*sizeof(float) );
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_gauss_filter_by_256,
                                         h_gauss_filter_by_256,
                                         9*sizeof(float) );
}

__host__
void Frame::applyGauss( )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = V7_WIDTH;
    grid.x  = getWidth() / V7_WIDTH;
    grid.y  = getHeight();

    filter_gauss_horiz_from_uchar
        <<<grid,block,0,_stream>>>
        ( _d_plane, _d_intermediate );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_intermediate, _d_smooth, GAUSS_TABLE );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_smooth, _d_intermediate, GAUSS_TABLE );

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_intermediate, _d_dx, GAUSS_DERIV );

    // possible to split into 2 streams
    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_smooth, _d_intermediate, GAUSS_TABLE );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_intermediate, _d_dy, GAUSS_DERIV );

    // necessary to merge into 1 stream
    compute_mag_l2
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag );

    // static const float kDefaultCannyThrLow      =  0.01f ;
    // static const float kDefaultCannyThrHigh     =  0.04f ;

    compute_map
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag, _d_map, 0.01f, 0.04f );

    edge_hysteresis
        <<<grid,block,0,_stream>>>
        ( _d_map, _d_edges );

#if 0
    // very costly printf-debugging
    debug_gauss
        <<<1,1,0,_stream>>>
        ( _d_smooth );
#endif

    cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void Frame::allocDevGaussianPlane( )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    void* ptr;
    const size_t w = getWidth();
    const size_t h = getHeight();
    size_t p;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    assert( p % _d_smooth.elemSize() == 0 );
    _d_smooth.data = (float*)ptr;
    _d_smooth.step = p;
    _d_smooth.cols = w;
    _d_smooth.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(int16_t), h );
    assert( p % _d_dx.elemSize() == 0 );
    _d_dx.data = (int16_t*)ptr;
    _d_dx.step = p;
    _d_dx.cols = w;
    _d_dx.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(int16_t), h );
    assert( p % _d_dy.elemSize() == 0 );
    _d_dy.data = (int16_t*)ptr;
    _d_dy.step = p;
    _d_dy.cols = w;
    _d_dy.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    assert( p % _d_intermediate.elemSize() == 0 );
    _d_intermediate.data = (float*)ptr;
    _d_intermediate.step = p;
    _d_intermediate.cols = w;
    _d_intermediate.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(uint32_t), h );
    assert( p % _d_mag.elemSize() == 0 );
    _d_mag.data = (uint32_t*)ptr;
    _d_mag.step = p;
    _d_mag.cols = w;
    _d_mag.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(unsigned char), h );
    assert( p % _d_map.elemSize() == 0 );
    _d_map.data = (unsigned char*)ptr;
    _d_map.step = p;
    _d_map.cols = w;
    _d_map.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(unsigned char), h );
    assert( p % _d_edges.elemSize() == 0 );
    _d_edges.data = (unsigned char*)ptr;
    _d_edges.step = p;
    _d_edges.cols = w;
    _d_edges.rows = h;

    POP_CUDA_MEMSET_ASYNC( _d_smooth.data,
                           0,
                           _d_smooth.step * _d_smooth.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_dx.data,
                           0,
                           _d_dx.step * _d_dx.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_dy.data,
                           0,
                           _d_dy.step * _d_dy.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_intermediate.data,
                           0,
                           _d_intermediate.step * _d_intermediate.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_mag.data,
                           0,
                           _d_mag.step * _d_mag.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_map.data,
                           0,
                           _d_map.step * _d_map.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_edges.data,
                           0,
                           _d_edges.step * _d_edges.rows,
                           _stream );

    cerr << "Leave " << __FUNCTION__ << endl;
}

}; // namespace popart


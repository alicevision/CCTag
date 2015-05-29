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

#define GAUSS_TABLE  0 // Gauss parameters
#define GAUSS_DERIV 16 // first derivative
__device__ __constant__ float d_gauss_filter[32];

__device__ __constant__ float d_gauss_filter_by_256[16];

__device__ __constant__ unsigned char d_thinning_lut[256];

__device__ __constant__ unsigned char d_thinning_lut_t[256];

#define V7_WIDTH    32

template <typename T>
__device__
inline
bool outOfBounds( int x, int y, const cv::cuda::PtrStepSz<T>& edges )
{
    return ( x < 0 || x >= edges.cols || y < 0 || y >= edges.rows );
}

template <typename T>
__device__
inline T d_abs( T value )
{
    return ( ( value < 0 ) ? -value : value );
}

template <typename T>
__device__
inline int d_sign( T value )
{
    return ( ( value < 0 ) ? -1 : 1 );
}

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
    int block_x = blockIdx.x * V7_WIDTH;
    int idx     = block_x + threadIdx.x;
    int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, dst ) ) return;

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

__device__
bool thinning_inner( const int idx, const int idy, cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst, bool first_run )
{
    uint8_t log = 0;

    if( src.ptr(idy)[idx] == 0 ) return false;

    if( idx >= 1 && idy >=1 && idx <= src.cols-2 && idy <= src.rows-2 ) {
        log |= ( src.ptr(idy-1)[idx-1] != 0 ) ? 0x80 : 0;
        log |= ( src.ptr(idy-1)[idx  ] != 0 ) ? 0x01 : 0;
        log |= ( src.ptr(idy-1)[idx+1] != 0 ) ? 0x02 : 0;
        log |= ( src.ptr(idy  )[idx+1] != 0 ) ? 0x04 : 0;
        log |= ( src.ptr(idy+1)[idx+1] != 0 ) ? 0x08 : 0;
        log |= ( src.ptr(idy+1)[idx  ] != 0 ) ? 0x10 : 0;
        log |= ( src.ptr(idy+1)[idx-1] != 0 ) ? 0x20 : 0;
        log |= ( src.ptr(idy  )[idx-1] != 0 ) ? 0x40 : 0;

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
    const int block_x = blockIdx.x * V7_WIDTH;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    thinning_inner( idx, idy, src, dst, true );
}

__global__
void thinning_and_store( cv::cuda::PtrStepSzb src, cv::cuda::PtrStepSzb dst, uint32_t* edgeCounter, uint32_t edgeMax, int2* edgeCoords )
{
    const int block_x = blockIdx.x * V7_WIDTH;
    const int idx     = block_x + threadIdx.x;
    const int idy     = blockIdx.y;

    bool keep = thinning_inner( idx, idy, src, dst, false );

    uint32_t mask = __ballot( keep );  // bitfield of warps with results
    uint32_t ct   = __popc( mask );    // horizontal reduce
    uint32_t leader = __ffs(mask) - 1; // the highest thread id with indicator==true
    uint32_t write_index;
    if( threadIdx.x == leader ) {
        // leader gets warp's offset from global value and increases it
        write_index = atomicAdd( edgeCounter, ct );
    }
    write_index = __shfl( write_index, leader ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index

    if( keep && write_index < edgeMax ) {
        edgeCoords[write_index] = make_int2( idx, idy );
    }
}

__device__
void updateXY(const float & dx, const float & dy, int & x, int & y,  float & e, int & stpX, int & stpY)
{
    float d = dy / dx;
    float a = d_abs( d );
    // stpX = ( dx < 0 ) ? -1 : ( dx == 0 ) ? 0 : 1;
    // stpY = ( dy < 0 ) ? -1 : ( dy == 0 ) ? 0 : 1;
    // stpX = ( dx < 0 ) ? -1 : 1;
    // stpY = ( dy < 0 ) ? -1 : 1;
    stpX = d_sign( dx );
    stpY = d_sign( dy );
    e   += a;
    x   += stpX;
    if( e >= 0.5 ) {
        y += stpY;
        e -= 1.0f;
    }
}

__device__
bool gradiant_descent_inner( int4&                  out_edge_info,
                             int2*                  d_edgelist,
                             uint32_t               edgeCount,
                             cv::cuda::PtrStepSzb   edges,
                             // int                    direction,
                             uint32_t               nmax,
                             cv::cuda::PtrStepSz16s d_dx,
                             cv::cuda::PtrStepSz16s d_dy,
                             int32_t                thrGradient )
{
    const int offset    = blockIdx.x * 32 + threadIdx.y;
    int direction = threadIdx.x == 0 ? 1 : -1;

    if( offset >= edgeCount ) return false;

    const int idx = d_edgelist[offset].x;
    const int idy = d_edgelist[offset].y;
    // const int block_x = blockIdx.x * V7_WIDTH;
    // const int idx     = block_x + threadIdx.x;
    // const int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, edges ) ) return false; // should never happen

    if( edges.ptr(idy)[idx] == 0 ) return false; // should never happen

    float  e     = 0.0f;
    float  dx    = direction * d_dx.ptr(idy)[idx];
    float  dy    = direction * d_dy.ptr(idy)[idx];
    const float  adx   = d_abs( dx );
    const float  ady   = d_abs( dy );
    size_t n     = 0;
    int    stpX  = 0;
    int    stpY  = 0;
    int    x     = idx;
    int    y     = idy;

    if( ady > adx ) {
        updateXY(dy,dx,y,x,e,stpY,stpX);
    } else {
        updateXY(dx,dy,x,y,e,stpX,stpY);
    }
    n += 1;
    if ( dx*dx+dy*dy > thrGradient ) {
        const float dxRef = dx;
        const float dyRef = dy;
        const float dx2 = d_dx.ptr(idy)[idx];
        const float dy2 = d_dy.ptr(idy)[idx];
        const float compdir = dx2*dxRef+dy2*dyRef;
        // dir = ( compdir < 0 ) ? -1 : 1;
        direction = d_sign( compdir );
        dx = direction * dx2;
        dy = direction * dy2;
    }
    if( ady > adx ) {
        updateXY(dy,dx,y,x,e,stpY,stpX);
    } else {
        updateXY(dx,dy,x,y,e,stpX,stpY);
    }
    n += 1;

    if( outOfBounds( x, y, edges ) ) return false;

    uint8_t ret = edges.ptr(y)[x];
    if( ret ) {
        out_edge_info = make_int4( idx, idy, x, y );
        return true;
    }
    
    while( n <= nmax ) {
        if( ady > adx ) {
            updateXY(dy,dx,y,x,e,stpY,stpX);
        } else {
            updateXY(dx,dy,x,y,e,stpX,stpY);
        }
        n += 1;

        if( outOfBounds( x, y, edges ) ) return false;

        ret = edges.ptr(y)[x];
        if( ret ) {
            out_edge_info = make_int4( idx, idy, x, y );
            return true;
        }

        if( ady > adx ) {
            if( outOfBounds( x, y - stpY, edges ) ) return false;

            ret = edges.ptr(y-stpY)[x];
            if( ret ) {
                out_edge_info = make_int4( idx, idy, x, y-stpY );
                return true;
            }
        } else {
            if( outOfBounds( x - stpX, y, edges ) ) return false;

            ret = edges.ptr(y)[x-stpX];
            if( ret ) {
                out_edge_info = make_int4( idx, idy, x-stpX, y );
                return true;
            }
        }
    }
    return false;
}

__global__
void gradiant_descent( int2*                  d_edgelist,
                       uint32_t               edgeCount,
                       int4*                  d_new_edgelist,
                       uint32_t*              d_new_edge_counter,
                       uint32_t               max_num_edges,
                       cv::cuda::PtrStepSzb   edges,
                       uint32_t               nmax,
                       cv::cuda::PtrStepSz16s d_dx,
                       cv::cuda::PtrStepSz16s d_dy,
                       int32_t                thrGradient )
{
    int4 out_edge_info;
    bool keep = gradiant_descent_inner( out_edge_info,
                                        d_edgelist,
                                        edgeCount,
                                        edges,
                                        nmax,
                                        d_dx,
                                        d_dy,
                                        thrGradient );

    uint32_t mask = __ballot( keep );  // bitfield of warps with results
    uint32_t ct   = __popc( mask );    // horizontal reduce
    uint32_t leader = __ffs(mask) - 1; // the highest thread id with indicator==true
    uint32_t write_index;
    if( threadIdx.x == leader ) {
        // leader gets warp's offset from global value and increases it
        write_index = atomicAdd( d_new_edge_counter, ct );
    }
    write_index = __shfl( write_index, leader ); // broadcast warp write index to all
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) ); // find own write index

    if( keep && write_index < max_num_edges ) {
        d_new_edgelist[write_index] = out_edge_info;
    }
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
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_thinning_lut,
                                         h_thinning_lut,
                                         256*sizeof(unsigned char) );
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_thinning_lut_t,
                                         h_thinning_lut_t,
                                         256*sizeof(unsigned char) );
}

__host__
void Frame::applyGauss( const cctag::Parameters & params )
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

    thinning
        <<<grid,block,0,_stream>>>
        ( _d_edges, cv::cuda::PtrStepSzb(_d_intermediate) );

    const uint32_t max_num_edges = params._maxEdges;
    uint32_t edge_counter = 0;
    POP_CUDA_MEMCPY_ASYNC( &_d_edge_counter, &edge_counter, sizeof(uint32_t), cudaMemcpyHostToDevice, _stream, true );

    thinning_and_store
        <<<grid,block,0,_stream>>>
        ( cv::cuda::PtrStepSzb(_d_intermediate), _d_edges, &_d_edge_counter, max_num_edges, _d_edgelist );

    POP_CUDA_MEMCPY_ASYNC( &edge_counter, &_d_edge_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, _stream, true );

    // Note: right here, Dynamic Parallelism would avoid blocking.
    cudaStreamSynchronize( _stream );
    // Try to figure out how that can be done with CMake.

    const uint32_t nmax          = params._distSearch;
    const int32_t  threshold     = params._thrGradientMagInVote;
    block.x = 2;
    block.y = 32;
    block.z = 0;
    grid.x  = edge_counter / 32 + ( edge_counter & 0x1f != 0 ? 1 : 0 );

    {
        uint32_t dummy = 0;
        POP_CUDA_MEMCPY_ASYNC( &_d_edge_counter, &dummy, sizeof(uint32_t), cudaMemcpyHostToDevice, _stream, true );
    }

    gradiant_descent
        <<<grid,block,0,_stream>>>
        ( _d_edgelist, edge_counter, _d_edgelist_2, &_d_edge_counter, max_num_edges, _d_edges, nmax, _d_dx, _d_dy, threshold ); // , _d_out_points );

    POP_CUDA_MEMCPY_ASYNC( &edge_counter, &_d_edge_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost,_stream, true );

    // Note: right here, Dynamic Parallelism would avoid blocking.
    cudaStreamSynchronize( _stream );
    // Try to figure out how that can be done with CMake.

    /*
     * After this:
     * - it is unclear whether a sweep over the entire edge plane
     *   is still an efficient choice, or whether it would be better
     *   to reduce the edge points into a flat array first.
     * - if reduction is the thing to do, it would be better to
     *   combine it with the second thinning step.
     *
     * cctagDetectionFromEdges takes as parameters:
     * - markers     : output, list of marker coordinates
     * - points      : a vector that contains all edge points (input)
     * - sourceView  : image at this scale (input)
     * - cannyGradX  : _d_dx (input)
     * - cannyGradY  : _d_dy (input)
     * - edgesMap    : _d_edges (input)
     * - frame       : running counter for video frames
     * - level       : this layer of the pyramid
     * - scale       : probably a coordinate multiplier
     * - params      : parameters
     *
     * calls vote
     *
     * vote takes as input:
     * - points     : a vector that contains all edge points (input)
     * - seeds      : unknown (output)
     * - edgesMap   : _d_edges (input)
     * - winners    : map of winners (output)
     * - cannyGradX : _d_dx (input)
     * - cannyGradY : _d_dy (input)
     * - params     : parameters
     *
     * calls gradientDirectionDescent
     *
     * gradientDirectionDescent takes as input
     * - canny       : _d_edges (input)
     * - p           : coordinate of one edge point (input)
     * - dir         : direction, 1 or -1
     * - nmax        : global parameter the distance from edge point for searching
     * - cannyGradX  : _d_dx
     * - cannyGradY  : _d_dy
     * - thrGradiant : global parameter for gradient thresholding
     * returns: NULL or a new point
     */
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


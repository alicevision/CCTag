/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "cctag/cuda/cctag_cuda_runtime.h"
#include "cctag/cuda/debug_macros.hpp"
#include "cctag/cuda/debug_image.h"

#include "cctag/cuda/frame.h"
#include "cctag/cuda/frameparam.h"
#include "cctag/cuda/clamp.h"
#include "cctag/cuda/assist.h"


namespace cctag
{

using namespace std;

__global__
void compute_mag_l1( DevPlane2D16s src_dx,
                     DevPlane2D16s src_dy,
                     DevPlane2D16s dst )
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
void compute_mag_l2( DevPlane2D16s src_dx,
                     DevPlane2D16s src_dy,
                     DevPlane2D16s dst )
{
    int block_x = blockIdx.x * 32;
    int idx     = block_x + threadIdx.x;
    int idy     = blockIdx.y;

    if( outOfBounds( idx, idy, dst ) ) return;

    int16_t dx = src_dx.ptr(idy)[idx];
    int16_t dy = src_dy.ptr(idy)[idx];
    // --- rintf( hypot ( ) ) --
    dx *= dx;
    dy *= dy;
    dst.ptr(idy)[idx] = __fsqrt_rn( (float)( dx + dy ) );
}

__global__
void compute_map( const DevPlane2D16s dx,
                  const DevPlane2D16s dy,
                  const DevPlane2D16s mag,
                  DevPlane2Db         map )
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

    if( magVal > tagParam.cannyThrLow_x_256 )
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
            edge_type = 1 + (uint8_t)(magVal > tagParam.cannyThrHigh_x_256);
        }
    }
    __syncthreads();

    assert( edge_type <= 2 );
    map.ptr(idy)[idx] = edge_type;
}

__host__
void Frame::applyMag( )
{
    dim3 block( 32, 1, 1 );
    dim3 grid( grid_divide( getWidth(), 32 ), getHeight(), 1 );

    // necessary to merge into 1 stream
    compute_mag_l2
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag );

    compute_map
        <<<grid,block,0,_stream>>>
        ( _d_dx, _d_dy, _d_mag, _d_map );

    /* block download until MAG and MAP are ready */
    cudaEventRecord( _download_ready_event.magmap, _stream );

#ifdef CCTAG_VISUAL_DEBUG
    cudaDeviceSynchronize();
    _h_plane.copyFrom( _d_plane );
    _h_dx.copyFrom( _d_dx );
    _h_dy.copyFrom( _d_dy );
    _h_mag.copyFrom( _d_mag );
    _h_debug_map.copyFrom( _d_map );
    std::ostringstream o0,o1,o2,o3,o4;
    o0 << "plane-" << _layer << "-cuda.pgm";
    o1 << "dx-" << _layer << "-cuda.pgm";
    o2 << "dy-" << _layer << "-cuda.pgm";
    o3 << "mag-" << _layer << "-cuda.pgm";
    o4 << "map-" << _layer << "-cuda.pgm";
    DebugImage::writePGM( o0.str(), _h_plane );
    DebugImage::writePGMscaled( o1.str(), _h_dx );
    DebugImage::writePGMscaled( o2.str(), _h_dy );
    DebugImage::writePGMscaled( o3.str(), _h_mag );
    DebugImage::writePGMscaled( o4.str(), _h_debug_map );
#endif // CCTAG_VISUAL_DEBUG
}

__host__
void Frame::applyMagDownload( )
{
    cudaStreamWaitEvent( _download_stream, _download_ready_event.magmap, 0 );

    _h_mag.copyFrom( _d_mag, _download_stream );

#ifdef DEBUG_WRITE_MAP_AS_PGM
    _h_debug_map.copyFrom( _d_map, _download_stream );
#endif // DEBUG_WRITE_MAP_AS_PGM
}

}; // namespace cctag


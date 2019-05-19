/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <fstream>
#include <cctag/cuda/cctag_cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"

namespace cctag
{

using namespace std;

/* These numbers are taken from Lilian's file cctag/fiter/cvRecode.cpp
 * Note that the array looks like because a __constant__ device array
 * with 2 dimensions is conceptually very problematic. The reason is
 * that the compiler pads each dimension separately, but there is no
 * way of asking about this padding (pitch, stepsize, whatever you
 * call it).
 * If the kernels should be multi-use, we need one array with two offsets.
 * Aligning to anything less than 16 floats is a bad idea.
 */

#undef NORMALIZE_GAUSS_VALUES

#ifdef NORMALIZE_GAUSS_VALUES
static const float sum_of_gauss_values = 0.000053390535453f +
                                         0.001768051711852f +
                                         0.021539279301849f +
                                         0.096532352630054f +
                                         0.159154943091895f +
                                         0.096532352630054f +
                                         0.021539279301849f +
                                         0.001768051711852f +
                                         0.000053390535453f;
static const float normalize_derived = 2.0f * ( 1.213061319425269f + 0.541341132946452f + 0.066653979229454f + 0.002683701023220f );
#endif // NORMALIZE_GAUSS_VALUES

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
    -0.002683701023220f,
    -0.066653979229454f,
    -0.541341132946452f,
    -1.213061319425269f,
    0.0f,
    1.213061319425269f,
    0.541341132946452f,
    0.066653979229454f,
    0.002683701023220f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f,
    0.0f
};

__device__ __constant__ float d_gauss_filter[32];
// __device__ __constant__ float d_gauss_filter_by_256[16];

template <class SrcType, class DestType>
__global__
void filter_gauss_horiz( cv::cuda::PtrStepSz<SrcType>  src,
                         cv::cuda::PtrStepSz<DestType> dst,
                         int                           filter,
                         float                         scale )
{
    const int idx     = blockIdx.x * 32 + threadIdx.x;
    const int idy     = blockIdx.y;
    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[filter + offset];

        int lookup = clamp( idx + offset - 4, src.cols );
        float val = src.ptr(idy)[lookup];
        out += ( val * g );
    }

    if( idy >= dst.rows ) return;
    if( idx*sizeof(DestType) >= src.step ) return;

    bool nix = ( idx >= dst.cols ) || ( idy >= dst.rows );
    out /= scale;
    dst.ptr(idy)[idx] = nix ? 0 : (DestType)out;
}

template <class SrcType, class DestType>
__global__
void filter_gauss_vert( cv::cuda::PtrStepSz<SrcType>  src,
                        cv::cuda::PtrStepSz<DestType> dst,
                        int                           filter,
                        float                         scale )
{
    const int idx     = blockIdx.x * 32 + threadIdx.x;
    const int idy     = blockIdx.y;
    float out = 0;

    if( idx*sizeof(SrcType) >= src.step ) return;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[filter + offset];

        int lookup = clamp( idy + offset - 4, src.rows );
        float val = src.ptr(lookup)[idx];
        out += ( val * g );
    }

    if( idy >= dst.rows ) return;

    bool nix = ( idx >= dst.cols ) || ( idy >= dst.rows );
    out /= scale;
    dst.ptr(idy)[idx] = nix ? 0 : (DestType)out;
}

__host__
void Frame::initGaussTable( )
{
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_gauss_filter,
                                         h_gauss_filter,
                                         32*sizeof(float) );
}

__host__
void Frame::applyGauss( const cctag::Parameters & params )
{
//    /*
//     * This is the original approach, following the explanation in cvRecode.
//     * However, the 1D tables that we use have already been convolved with
//     * an initial Gauss step, and give the wrong results. So, the first sweep
//     * must be removed.
//     * If the goal was to smoothe the picture, that would be a mistake,
//     * because multiple sweeps extend the range of the filter and bring the
//     * result closer to a globally applied Gaussian filter. However, for CCTag,
//     * this is just a strengthening of the edge signal of a single pixel in its
//     * surrounding area. The far distant pixels don't matter.
//     */
//
//    filter_gauss_horiz_from_uchar<<<grid,block,0,_stream>>>( _d_plane, _d_intermediate, sum_of_gauss_values );
//    filter_gauss_vert<<<grid,block,0,_stream>>>( _d_intermediate, _d_smooth, GAUSS_TABLE, sum_of_gauss_values );
//    filter_gauss_vert<<<grid,block,0,_stream>>>( _d_smooth, _d_intermediate, GAUSS_TABLE, 1.0f );
//    filter_gauss_horiz<<<grid,block,0,_stream>>>( _d_intermediate, _d_debug_dx, GAUSS_DERIV, 1.0f );
//    filter_gauss_horiz<<<grid,block,0,_stream>>>( _d_smooth, _d_intermediate, GAUSS_TABLE, 1.0f );
//    filter_gauss_vert<<<grid,block,0,_stream>>>( _d_intermediate, _d_dy, GAUSS_DERIV, 1.0f );
//

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = ( getWidth() / 32 )  + ( getWidth() % 32 == 0 ? 0 : 1 );
    grid.y  = getHeight();
    assert( grid.x > 0 && grid.y > 0 && grid.z > 0 );
    assert( block.x > 0 && block.y > 0 && block.z > 0 );

#ifdef NORMALIZE_GAUSS_VALUES
    const float normalize   = sum_of_gauss_values;
    const float normalize_d = normalize_derived;
#else // NORMALIZE_GAUSS_VALUES
    const float normalize   = 1.0f;
    const float normalize_d = 1.0f;
#endif // NORMALIZE_GAUSS_VALUES
    /*
     * Vertical sweep for DX computation: use Gaussian table
     */
    filter_gauss_vert<<<grid,block,0,_stream>>>( _d_plane, _d_intermediate, GAUSS_TABLE, normalize );

    /*
     * Compute DX
     */
    filter_gauss_horiz<<<grid,block,0,_stream>>>( _d_intermediate, _d_dx, GAUSS_DERIV, normalize_d );

    /*
     * Compute DY
     */
    filter_gauss_vert <<<grid,block,0,_stream>>>( _d_plane, _d_intermediate, GAUSS_DERIV, normalize_d );

    /*
     * Horizontal sweep for DY computation: use Gaussian table
     */
    filter_gauss_horiz<<<grid,block,0,_stream>>>( _d_intermediate, _d_dy, GAUSS_TABLE, normalize );

    /* generate event when DX and DY are ready */
    cudaEventRecord( _download_ready_event.dxdy, _stream );
}

__host__
void Frame::applyGaussDownload( )
{
    cudaError_t err;

    /* block download until DX and DY are ready */
    err = cudaStreamWaitEvent( _download_stream, _download_ready_event.dxdy, 0 );
    POP_CUDA_FATAL_TEST( err, "Cannot wait for download stream event: " );

    // After these linking operations, dx and dy are created for
    // all edge points and we can copy them to the host

    cudaMemcpy2DAsync( _h_dx.data, _h_dx.step,
                       _d_dx.data, _d_dx.step,
                       _d_dx.cols * sizeof(int16_t),
                       _d_dx.rows,
                       cudaMemcpyDeviceToHost, _download_stream );

    cudaMemcpy2DAsync( _h_dy.data, _h_dy.step,
                       _d_dy.data, _d_dy.step,
                       _d_dy.cols * sizeof(int16_t),
                       _d_dy.rows,
                       cudaMemcpyDeviceToHost, _download_stream );
}

}; // namespace cctag


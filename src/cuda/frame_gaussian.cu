// #include <iostream>
// #include <limits>
#include <cuda_runtime.h>
// #include <stdio.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"

namespace popart
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

static const float sum_of_gauss_values = 0.000053390535453f +
                                         0.001768051711852f +
                                         0.021539279301849f +
                                         0.096532352630054f +
                                         0.159154943091895f +
                                         0.096532352630054f +
                                         0.021539279301849f +
                                         0.001768051711852f +
                                         0.000053390535453f;

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
__device__ __constant__ float d_gauss_filter_by_256[16];

template <class DestType>
__global__
void filter_gauss_horiz( cv::cuda::PtrStepSzf          src,
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
    if( idx*sizeof(float) >= src.step ) return;

    bool nix = ( idx >= dst.cols ) || ( idy >= dst.rows );
    out /= scale;
    dst.ptr(idy)[idx] = nix ? 0 : (DestType)out;
}

template <class DestType>
__global__
void filter_gauss_vert( cv::cuda::PtrStepSzf          src,
                        cv::cuda::PtrStepSz<DestType> dst,
                        int                           filter,
                        float                         scale )
{
    const int idx     = blockIdx.x * 32 + threadIdx.x;
    const int idy     = blockIdx.y;
    float out = 0;

    if( idx*sizeof(float) >= src.step ) return;

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

__global__
void filter_gauss_horiz_from_uchar( cv::cuda::PtrStepSzb src,
                                    cv::cuda::PtrStepSzf dst,
                                    float                scale )
{
    const int idx     = blockIdx.x * 32 + threadIdx.x;
    const int idy     = blockIdx.y;
    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[offset];

        int lookup = clamp( idx + offset - 4, src.cols );
        float val = src.ptr(idy)[lookup];
        out += ( val * g );
    }

    if( idy >= dst.rows ) return;
    if( idx * sizeof(float) >= dst.step ) return;

    bool nix = ( idx >= dst.cols ) || ( idy >= dst.rows );
    out /= scale;
    dst.ptr(idy)[idx] = nix ? 0 : out;
}

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
void Frame::applyGauss( const cctag::Parameters & params )
{
    // cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = 32;
    grid.x  = ( getWidth() / 32 )  + ( getWidth() % 32 == 0 ? 0 : 1 );
    grid.y  = getHeight();

    filter_gauss_horiz_from_uchar
        <<<grid,block,0,_stream>>>
        ( _d_plane, _d_intermediate, sum_of_gauss_values );
    POP_CHK_CALL_IFSYNC;

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_intermediate, _d_smooth, GAUSS_TABLE, sum_of_gauss_values );
    POP_CHK_CALL_IFSYNC;

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_smooth, _d_intermediate, GAUSS_TABLE, sum_of_gauss_values );
    POP_CHK_CALL_IFSYNC;

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_intermediate, _d_dx, GAUSS_DERIV, 1.0f );
    POP_CHK_CALL_IFSYNC;

    // possible to split into 2 streams
    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_smooth, _d_intermediate, GAUSS_TABLE, sum_of_gauss_values );
    POP_CHK_CALL_IFSYNC;

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_intermediate, _d_dy, GAUSS_DERIV, 1.0f );
    POP_CHK_CALL_IFSYNC;

#ifdef EDGE_LINKING_HOST_SIDE
    // After these linking operations, dx and dy are created for
    // all edge points and we can copy them to the host

    POP_CUDA_MEMCPY_2D_ASYNC( _h_dx.data, _h_dx.step,
                              _d_dx.data, _d_dx.step,
                              _d_dx.cols * sizeof(int16_t),
                              _d_dx.rows,
                              cudaMemcpyDeviceToHost, _stream );

    POP_CUDA_MEMCPY_2D_ASYNC( _h_dy.data, _h_dy.step,
                              _d_dy.data, _d_dy.step,
                              _d_dy.cols * sizeof(int16_t),
                              _d_dy.rows,
                              cudaMemcpyDeviceToHost, _stream );
    POP_CHK_CALL_IFSYNC;
#endif // EDGE_LINKING_HOST_SIDE

    // cerr << "Leave " << __FUNCTION__ << endl;
}
}; // namespace popart


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

/* these numbers are taken from Lilian's file cctag/fiter/cvRecode.cpp */

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

__global__
void filter_gauss_horiz( cv::cuda::PtrStepSzf src,
                         cv::cuda::PtrStepSzf dst,
                         int                  filter )
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
    dst.ptr(block_y)[idx] = nix ? 0 : out;
}

__global__
void filter_gauss_vert( cv::cuda::PtrStepSzf src,
                        cv::cuda::PtrStepSzf dst,
                        int                  filter )
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
    dst.ptr(idy)[idx] = nix ? 0 : out;
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
        float g  = d_gauss_filter_by_256[offset];

        idx = clamp( block_x + threadIdx.x - offset - 4, src.cols );
        float val = src.ptr(block_y)[idx];
        out += ( val * g );
    }

    if( block_y >= dst.rows ) return;
    if( idx * sizeof(float) >= dst.step ) return;

    bool nix = ( block_x + threadIdx.x >= dst.cols ) || ( block_y >= dst.rows );
    dst.ptr(block_y)[idx] = nix ? 0 : out;
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

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_smooth, _d_intermediate, GAUSS_TABLE );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_intermediate, _d_dy, GAUSS_DERIV );

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

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    _d_dx.data = (float*)ptr;
    _d_dx.step = p;
    _d_dx.cols = w;
    _d_dx.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    _d_dy.data = (float*)ptr;
    _d_dy.step = p;
    _d_dy.cols = w;
    _d_dy.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    _d_intermediate.data = (float*)ptr;
    _d_intermediate.step = p;
    _d_intermediate.cols = w;
    _d_intermediate.rows = h;

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

    cerr << "Leave " << __FUNCTION__ << endl;
}

}; // namespace popart


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

static const float h_gauss_filter[9] =
{
    0.002683701023220,
    0.066653979229454,
    0.541341132946452,
    1.213061319425269,
    0,
    -1.213061319425269,
    -0.541341132946452,
    -0.066653979229454,
    -0.002683701023220
};

__device__ __constant__ float d_gauss_filter[16];
__device__ __constant__ float d_gauss_filter_by_256[16];

#define V7_WIDTH    32

__global__
void filter_gauss_horiz( cv::cuda::PtrStepSzf src,
                         cv::cuda::PtrStepSzf dst )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;
    int idx;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[offset];

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
                        cv::cuda::PtrStepSzf dst )
{
    const int block_x = blockIdx.x * V7_WIDTH;
    const int block_y = blockIdx.y;
    const int idx     = block_x + threadIdx.x;
    int idy;

    if( idx*sizeof(float) >= src.step ) return;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[offset];

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

__host__
void Frame::initGaussTable( )
{
    float h_gauss_filter_by_256[9];
    for( int i=0; i<9; i++ ) {
        h_gauss_filter_by_256[i] = h_gauss_filter[i] / 256.0f;
    }

    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_gauss_filter,
                                         h_gauss_filter,
                                         9*sizeof(float) );
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
        ( _d_plane, _d_gaussian_intermediate );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate, _d_gaussian );

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_gaussian, _d_gaussian_intermediate );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate, _d_gaussian );

    filter_gauss_horiz
        <<<grid,block,0,_stream>>>
        ( _d_gaussian, _d_gaussian_intermediate );

    filter_gauss_vert
        <<<grid,block,0,_stream>>>
        ( _d_gaussian_intermediate, _d_gaussian );

    debug_gauss
        <<<1,1,0,_stream>>>
        ( _d_gaussian );

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
    assert( p % _d_gaussian.elemSize() == 0 );
    _d_gaussian.data = (float*)ptr;
    _d_gaussian.step = p;
    _d_gaussian.cols = w;
    _d_gaussian.rows = h;

    POP_CUDA_MEMSET_ASYNC( _d_gaussian.data,
                           0,
                           _d_gaussian.step * _d_gaussian.rows,
                           _stream );

    cerr << "    allocated _d_gaussian with "
         << "(" << w << "," << h << ") pitch " << _d_gaussian.step
         << "(" << p << " bytes)" << endl;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    _d_gaussian_intermediate.data = (float*)ptr;
    _d_gaussian_intermediate.step = p;
    _d_gaussian_intermediate.cols = w;
    _d_gaussian_intermediate.rows = h;

    POP_CUDA_MEMSET_ASYNC( _d_gaussian_intermediate.data,
                           0,
                           _d_gaussian_intermediate.step * _d_gaussian_intermediate.rows,
                           _stream );

    cerr << "    allocated intermediat with "
         << "(" << w << "," << h << ") pitch " << _d_gaussian_intermediate.step
         << "(" << p << " bytes)" << endl;

    cerr << "Leave " << __FUNCTION__ << endl;
}

}; // namespace popart


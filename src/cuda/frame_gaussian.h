#pragma once

namespace popart
{

/*
 * This is actually a code file, to be included into frame.cu
 */

/* these numbers are taken from Lilian's file cctag/fiter/cvRecode.cpp */

const float h_gauss_filter[9] =
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
        float val = src.data[ block_y * src.step + idx ];
        out += ( val * g );
    }

    if( block_y >= dst.rows ) return;
    if( idx     >= dst.step ) return;

    bool nix = ( block_x + threadIdx.x >= dst.cols ) || ( block_y >= dst.rows );
    dst.data[ block_y * dst.step + idx ] = nix ? 0 : out;
}

__global__
void filter_gauss_vert( cv::cuda::PtrStepSzf src,
                        cv::cuda::PtrStepSzf dst )
{
    const int block_x = blockIdx.x * V7_WIDTH;
    const int block_y = blockIdx.y;
    const int idx     = block_x + threadIdx.x;
    int idy;

    if( idx >= src.step ) return;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[offset];

        idy = clamp( block_y - offset - 4, src.rows );
        float val = src.data[ idy * src.step + idx ];
        out += ( val * g );
    }

    if( idy >= dst.rows ) return;
    if( idx >= dst.step  ) return;

    bool nix = ( idx >= dst.cols );
    dst.data[ idy * dst.step + idx ] = nix ? 0 : out;
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
        float val = src.data[ block_y * src.step + idx ];
        out += ( val * g );
    }

    if( block_y >= dst.rows ) return;
    if( idx     >= dst.step ) return;

    bool nix = ( block_x + threadIdx.x >= dst.cols ) || ( block_y >= dst.rows );
    dst.data[ block_y * dst.step + idx ] = nix ? 0 : out;
}

}; // namespace popart


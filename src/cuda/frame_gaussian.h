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
// #define V7_RANGE    4 // RANGES from 1 to 12 are possible
// #define V7_GAUSS_BASE   ( GAUSS_ONE_SIDE_RANGE - V7_RANGE )
// #define V7_FILTERSIZE   ( V7_RANGE + 1        + V7_RANGE )
// #define V7_READ_RANGE   ( V7_RANGE + V7_WIDTH + V7_RANGE )
// #define V7_LEVELS       _levels

__global__
void filter_gauss_horiz( float*   src_data,
                         float*   dst_data,
                         uint32_t width,
                         uint32_t pitch,
                         uint32_t height )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;
    int idx;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[offset];

        idx = clamp( block_x + threadIdx.x - offset, width );
        float val = src_data[ block_y * pitch + idx ];
        out += ( val * g );
    }

    if( block_y >= height ) return;
    if( idx     >= pitch ) return;

    bool nix = ( block_x + threadIdx.x >= width ) || ( block_y >= height );
    dst_data[ block_y * pitch + idx ] = nix ? 0 : out;
}

__global__
void filter_gauss_vert( float*   src_data,
                           float*   dst_data,
                           uint32_t width,
                           uint32_t pitch,
                           uint32_t height )
{
    const int block_x = blockIdx.x * V7_WIDTH;
    const int block_y = blockIdx.y;
    const int idx     = block_x + threadIdx.x;
    int idy;

    if( idx >= pitch ) return;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter[offset];

        idy = clamp( block_y - offset, height );
        float val = src_data[ idy * pitch + idx ];
        out += ( val * g );
    }

    if( idy >= height ) return;
    if( idx >= pitch  ) return;

    bool nix = ( idx >= width );
    dst_data[ idy * pitch + idx ] = nix ? 0 : out;
}

__global__
void filter_gauss_horiz_from_uchar( unsigned char*   src_data,
                                    float*           dst_data,
                                    uint32_t         width,
                                    uint32_t         pitch,
                                    uint32_t         height )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;
    int idx;

    float out = 0;

    for( int offset = 0; offset<9; offset++ ) {
        float g  = d_gauss_filter_by_256[offset];

        idx = clamp( block_x + threadIdx.x - offset, width );
        float val = src_data[ block_y * pitch + idx ];
        out += ( val * g );
    }

    if( block_y >= height ) return;
    if( idx     >= pitch ) return;

    bool nix = ( block_x + threadIdx.x >= width ) || ( block_y >= height );
    dst_data[ block_y * pitch + idx ] = nix ? 0 : out;
}

}; // namespace popart


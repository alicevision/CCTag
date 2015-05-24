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

void Frame::initGaussTable( )
{
    POP_CUDA_MEMCPY_HOST_TO_SYMBOL_SYNC( d_gauss_filter, h_gauss_filter, 9*sizeof(float) );
}

}; // namespace popart


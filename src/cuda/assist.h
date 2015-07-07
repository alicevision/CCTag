#pragma once

#include <cuda_runtime.h>
// #include <assert.h>
// #include <string>

#include <opencv2/core/cuda_types.hpp>

namespace popart
{

template <typename T>
__device__ __host__
inline bool outOfBounds( int x, int y, const cv::cuda::PtrStepSz<T>& edges )
{
    return ( x < 0 || x >= edges.cols || y < 0 || y >= edges.rows );
}

template <typename T>
__device__ __host__
inline bool outOfBounds( int2 coord, const cv::cuda::PtrStepSz<T>& edges )
{
    return ( coord.x < 0 || coord.x >= edges.cols || coord.y < 0 || coord.y >= edges.rows );
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

__device__
inline bool reduceAND_32x32( bool cnt )
{
    __shared__ int reduce_array[32];

    int cnt_row = __any( cnt );
    if( threadIdx.x == 0 ) {
        reduce_array[threadIdx.y] = cnt_row;
    }
    __syncthreads();
    if( threadIdx.y == 0 ) {
        int cnt_col = __any( reduce_array[threadIdx.x] );
        if( threadIdx.x == 0 ) {
            reduce_array[0] = cnt_col;
        }
    }
    __syncthreads();
    cnt_row = reduce_array[0];
    return ( cnt_row != 0 );
}

}; // namespace popart


#pragma once

#include <cuda_runtime.h>

using namespace std;

/* This computation is needed very frequently when a dim3 grid block is
 * initialized. It ensure that the tail is not forgotten.
 */
__device__ __host__
inline int grid_divide( int size, int divider )
{
    return size / divider + ( size % divider != 0 ? 1 : 0 );
}

template <typename T>
__device__
inline T d_abs( T value )
{
    return ( ( value < 0 ) ? -value : value );
}

// this is usually not needed because CUDA has copysign()
template <typename T>
__device__
inline int d_sign( T value )
{
    return ( ( value < 0 ) ? -1 : 1 );
}

__device__
inline
bool reduce_OR_32x32( bool cnt )
{
    __shared__ int reduce_array[32];

    int cnt_row = ::__any( (int)cnt );
    if( threadIdx.x == 0 ) {
        reduce_array[threadIdx.y] = cnt_row;
    }
    __syncthreads();
    if( threadIdx.y == 0 ) {
        int cnt_col = ::__any( reduce_array[threadIdx.x] );
        if( threadIdx.x == 0 ) {
            reduce_array[0] = cnt_col;
        }
    }
    __syncthreads();
    cnt_row = reduce_array[0];
    return ( cnt_row != 0 );
}


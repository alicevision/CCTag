/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/core/cuda_types.hpp>

using namespace std;

std::ostream& operator<<( std::ostream& ostr, const dim3& p );

namespace popart
{

/* This computation is needed very frequently when a dim3 grid block is
 * initialized. It ensure that the tail is not forgotten.
 */
__device__ __host__
inline int grid_divide( int size, int divider )
{
    return size / divider + ( size % divider != 0 ? 1 : 0 );
}

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
inline
bool reduce_OR_32x32( bool cnt )
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


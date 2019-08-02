/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <iostream>
#include <cctag/cuda/cctag_cuda_runtime.h>
#include <opencv2/core/cuda_types.hpp>

namespace cctag {

std::ostream& operator<<( std::ostream& ostr, const dim3& p );

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

#ifdef CCTAG_HAVE_SHFL_DOWN_SYNC
template<typename T> __device__ inline T shuffle     ( T variable, int src   ) { return __shfl_sync     ( 0xffffffff, variable, src   ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta ) { return __shfl_up_sync  ( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta ) { return __shfl_down_sync( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor_sync ( 0xffffffff, variable, delta ); }
__device__ inline unsigned int ballot( unsigned int pred ) { return __ballot_sync   ( 0xffffffff, pred ); }
__device__ inline int any            ( unsigned int pred ) { return __any_sync      ( 0xffffffff, pred ); }
__device__ inline int all            ( unsigned int pred ) { return __all_sync      ( 0xffffffff, pred ); }

template<typename T> __device__ inline T shuffle     ( T variable, int src  , int ws ) { return __shfl_sync     ( 0xffffffff, variable, src  , ws ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up_sync  ( 0xffffffff, variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down_sync( 0xffffffff, variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor_sync ( 0xffffffff, variable, delta, ws ); }
#else
template<typename T> __device__ inline T shuffle     ( T variable, int src   ) { return __shfl     ( variable, src   ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta ) { return __shfl_up  ( variable, delta ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta ) { return __shfl_down( variable, delta ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor ( variable, delta ); }
__device__ inline unsigned int ballot( unsigned int pred ) { return __ballot   ( pred ); }
__device__ inline int any            ( unsigned int pred ) { return __any      ( pred ); }
__device__ inline int all            ( unsigned int pred ) { return __all      ( pred ); }

template<typename T> __device__ inline T shuffle     ( T variable, int src  , int ws ) { return __shfl     ( variable, src  , ws ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up  ( variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down( variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor ( variable, delta, ws ); }
#endif

__device__
inline
bool reduce_OR_32x32( bool cnt )
{
    __shared__ int reduce_array[32];

    int cnt_row = cctag::any( cnt );
    if( threadIdx.x == 0 ) {
        reduce_array[threadIdx.y] = cnt_row;
    }
    __syncthreads();
    if( threadIdx.y == 0 ) {
        int cnt_col = cctag::any( reduce_array[threadIdx.x] );
        if( threadIdx.x == 0 ) {
            reduce_array[0] = cnt_col;
        }
    }
    __syncthreads();
    cnt_row = reduce_array[0];
    return ( cnt_row != 0 );
}

}; // namespace cctag


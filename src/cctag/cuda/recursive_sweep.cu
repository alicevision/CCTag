/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/cctag_cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"
#include "assist.h"

using namespace std;

namespace cctag
{

namespace recursive_sweep
{

template<typename T>
__device__
inline
T get( const cv::cuda::PtrStepSz<T> img, const int idx, const int idy )
{
    return img.ptr( clamp( idy, img.rows ) )[ clamp( idx, img.cols ) ];
}

class EdgeExpanderProcessor
{
public:
    __device__
    inline
    bool check( cv::cuda::PtrStepSzb img, const int idx, const int idy )
    {
        if( idx == 0 || idy == 0 || idx >= img.cols-1 || idy >= img.rows-1 ) return false;

        uint8_t val = get( img, idx, idy );

        if( val != 1 ) return false;

        int n;
        n  = ( get( img, idx-1, idy-1 ) == 2 );
        n += ( get( img, idx  , idy-1 ) == 2 );
        n += ( get( img, idx+1, idy-1 ) == 2 );
        n += ( get( img, idx-1, idy   ) == 2 );
        n += ( get( img, idx+1, idy   ) == 2 );
        n += ( get( img, idx-1, idy+1 ) == 2 );
        n += ( get( img, idx  , idy+1 ) == 2 );
        n += ( get( img, idx+1, idy+1 ) == 2 );

        if( n == 0 ) return false;

        img.ptr(idy)[idx] = 2;

        return true;
    }
};

class ConnectedComponentProcessor
{
public:
    __device__
    inline
    bool check( cv::cuda::PtrStepSz32s img, const int idx, const int idy )
    {
        if( outOfBounds( idx, idy, img ) ) return false;

        uint8_t val = get( img, idx, idy );

        if( val == 0 ) return false;

        int oldval = val;
        val = max( val, get( img, idx-1, idy-1 ) );
        val = max( val, get( img, idx  , idy-1 ) );
        val = max( val, get( img, idx+1, idy-1 ) );
        val = max( val, get( img, idx-1, idy   ) );
        val = max( val, get( img, idx+1, idy   ) );
        val = max( val, get( img, idx-1, idy+1 ) );
        val = max( val, get( img, idx  , idy+1 ) );
        val = max( val, get( img, idx+1, idy+1 ) );

        if( val == oldval ) return false;

        img.ptr(idy)[idx] = val;

        return true;
    }
};

// HYST    4 x 32 -> >10
// HYST    2 x 32 -> >11.5
// HYST   16 x 16 -> 9.380 - 10.426
// HYST    8 x 32 -> 9.6 - 9.9
// HYST   32 x 32 -> 9.460 - 9.793
// HYST    8 x  8 -> 9.7 - 10.16
#define HYST_H   32
#define HYST_W   32

#if HYST_W < HYST_H
#error The code requires W<=32 and H<=W
#endif

template<typename T, class Processor>
__device__
bool single_block_loop( cv::cuda::PtrStepSz<T> img )
{
    __shared__ bool continuation[HYST_H];
    const int       idx  = blockIdx.x * HYST_W + threadIdx.x;
    const int       idy  = blockIdx.y * HYST_H + threadIdx.y;
    bool            again = true;
    bool            nothing_changed = true;

    Processor proc;
    while( again ) {
        bool mark      = proc.check( img, idx, idy );
        bool any_marks = cctag::any( mark );
        if( threadIdx.x == 0 ) continuation[threadIdx.y] = any_marks;
        __syncthreads();
        mark = threadIdx.x < HYST_H ? continuation[threadIdx.x] : false;
        __syncthreads();
        again = cctag::any( mark );
        if( again ) nothing_changed = false;
    }

    return nothing_changed;
}

template<typename T, class Processor>
__global__
void single_sweep( cv::cuda::PtrStepSz<T> img, int* counter )
{
    bool nothing_changed = single_block_loop<T,Processor>( img );
    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        if( nothing_changed ) {
            atomicSub( counter, 1 );
        }
#if 0
        int am_i_last = atomicSub( &edge_hysteresis_block_counter.x, 1 );
        if( am_i_last == 1 ) {
            int active_blocks = edge_hysteresis_block_counter.y;
            edge_hysteresis_block_again = active_blocks;
            // only useful for Dynamic Parallelism
        }
#endif
    }
}

template<typename T, class Processor>
__host__
void sweep_no_dynamic_parallelism( cv::cuda::PtrStepSz<T>& img,
                                   int*                    dev_counter,
                                   cudaStream_t            stream )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = HYST_W;
    block.y = HYST_H;
    grid.x  = grid_divide( img.cols, HYST_W );
    grid.y  = grid_divide( img.rows, HYST_H );

    int host_counter;
    do
    {
        host_counter = grid.x * grid.y;
        POP_CUDA_MEMCPY_TO_DEVICE_ASYNC( dev_counter,
                                         &host_counter,
                                         sizeof(int), stream );
        single_sweep<T,Processor>
            <<<grid,block,0,stream>>>
            ( img,
              dev_counter );
        POP_CHK_CALL_IFSYNC;

        POP_CUDA_MEMCPY_TO_HOST_ASYNC( &host_counter,
                                       dev_counter,
                                       sizeof(int), stream );
        POP_CUDA_SYNC( stream );
        cerr << "  Still active blocks: " << host_counter << endl;
    }
    while( host_counter > 0 );

    cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void expandEdges( cv::cuda::PtrStepSzb& img, int* dev_counter, cudaStream_t stream )
{
    recursive_sweep::sweep_no_dynamic_parallelism
        <uint8_t,EdgeExpanderProcessor>
        ( img, dev_counter, stream );
}

__host__
void connectComponents( cv::cuda::PtrStepSz32s& img, int* dev_counter, cudaStream_t stream )
{
    recursive_sweep::sweep_no_dynamic_parallelism
        <int,ConnectedComponentProcessor>
        ( img, dev_counter, stream );
}

}; // namespace recursive_sweep
}; // namespace cctag


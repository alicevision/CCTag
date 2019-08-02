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

namespace cctag
{

using namespace std;

namespace hysteresis
{
#define HYST_H   32
#define HYST_W   32

#if HYST_W < HYST_H
#error The code requires W<=32 and H<=W
#endif

__shared__ volatile uint8_t array[HYST_H+2][4*(HYST_W+2)];

__device__
inline
uint32_t get( cv::cuda::PtrStepSz32u img, const int idx, const int idy )
{
    if( idx < 0 || idy < 0 || idx >= img.cols || idy >= img.rows ) {
        return 0;
    }
    return img.ptr(idy)[idx];
}

__device__
void load( cv::cuda::PtrStepSz32u img )
{
    const int srcidx = blockIdx.x * HYST_W + threadIdx.x;
    const int srcidy = blockIdx.y * HYST_H + threadIdx.y;
    uint32_t v;

    volatile uint32_t* load_line;
    load_line = reinterpret_cast<volatile uint32_t*>(&array[threadIdx.y][0]);

    v = get( img, srcidx-1, srcidy-1 );
    load_line[threadIdx.x  ] = v;

    if( threadIdx.x >= HYST_W - 2 ) {
        v = get( img, srcidx+1, srcidy-1 );
        load_line[threadIdx.x+2] = v;
    }
    __syncthreads();

    if( threadIdx.y >= HYST_H - 2 ) {
        load_line = reinterpret_cast<volatile uint32_t*>(&array[threadIdx.y+2][0]);

        v = get( img, srcidx-1, srcidy+1 );
        load_line[threadIdx.x  ] = v;

        if( threadIdx.x >= HYST_W - 2 ) {
            v = get( img, srcidx+1, srcidy+1 );
            load_line[threadIdx.x+2] = v;
        }
    }
    __syncthreads();
}

__device__
void store( cv::cuda::PtrStepSz32u img )
{
    const int dstidx  = blockIdx.x * HYST_W + threadIdx.x;
    const int dstidy  = blockIdx.y * HYST_H + threadIdx.y;

    if( dstidx*sizeof(uint32_t) < img.step && dstidy < img.rows ) {
        // volatile uint32_t* shared_line = reinterpret_cast<volatile uint32_t*>(array[threadIdx.y+1]);
        volatile uint32_t* shared_line = reinterpret_cast<volatile uint32_t*>(&array[threadIdx.y+1][0]);
        uint32_t val = shared_line[threadIdx.x+1];

        img.ptr(dstidy)[dstidx] = val;
    }
}

__device__
inline
bool update_edge_pixel( int y, int x )
{
    bool something_changed = false;

    union {
        uint8_t  b[12];
        uint3    i;
    } val[3];

    val[0].i = make_uint3( reinterpret_cast<volatile uint32_t*>( &array[y  ][x] )[0],
                           reinterpret_cast<volatile uint32_t*>( &array[y  ][x] )[1],
                           reinterpret_cast<volatile uint32_t*>( &array[y  ][x] )[2] );
    val[1].i = make_uint3( reinterpret_cast<volatile uint32_t*>( &array[y+1][x] )[0],
                           reinterpret_cast<volatile uint32_t*>( &array[y+1][x] )[1],
                           reinterpret_cast<volatile uint32_t*>( &array[y+1][x] )[2] );
    val[2].i = make_uint3( reinterpret_cast<volatile uint32_t*>( &array[y+1][x] )[0],
                           reinterpret_cast<volatile uint32_t*>( &array[y+2][x] )[1],
                           reinterpret_cast<volatile uint32_t*>( &array[y+2][x] )[2] );

    for( int i=0; i<4; i++ ) {
        bool inc = false;
        bool dec = false;

        if( val[1].b[4+i] == 1 ) {
            inc = ( val[0].b[3+i] == 2 || val[0].b[4+i] == 2 || val[0].b[5+i] == 2 ||
                    val[1].b[3+i] == 2 ||                       val[1].b[5+i] == 2 ||
                    val[2].b[3+i] == 2 || val[2].b[4+i] == 2 || val[2].b[5+i] == 2 );
            dec = ( val[0].b[3+i] == 0 && val[0].b[4+i] == 0 && val[0].b[5+i] == 0 &&
                    val[1].b[3+i] == 0 &&                       val[1].b[5+i] == 0 &&
                    val[2].b[3+i] == 0 && val[2].b[4+i] == 0 && val[2].b[5+i] == 0 );
            val[1].b[4+i] = inc ? 2 : dec ? 0 : 1 ;
        }
        __syncthreads();

        something_changed |= inc;
        something_changed |= dec;
    }
    reinterpret_cast<volatile uint32_t*>( &array[y+1][x] )[1] = val[1].i.y;

    return something_changed;
}

__device__
bool edge_block_loop( )
{
    __shared__ volatile bool continuation[HYST_H];
    bool            again = true;
    bool            something_changed = false;
    int debug_inner_loop_count = 0;

    // DEBUG NOTE:
    // updating in the inner loop works correctly
    // but the outer loop repeats exactly changes in the inner loop, in particular on
    // x values of 0, 1 or 2
    // WHY ?

    while( again ) { // && debug_inner_loop_count < 10 ) {
        assert( debug_inner_loop_count <= HYST_W*HYST_H );

        bool mark = update_edge_pixel( threadIdx.y, threadIdx.x*4 );

        /* every row checks whether any pixel has been changed */
        bool line_changed = cctag::any( mark );

#if 0
        /* the first thread of each row write the result to continuation[] */
        if( threadIdx.x == 0 ) continuation[threadIdx.y] = line_changed;

        /* make sure all updated pixel are written back to
         * shared memory before continuation[] is modified.
         * This is supposedly redundant with __syncthreads() */
        __threadfence_block();

        /* wait for all rows to fulfill the operation (and to assure that
         * results in continuation[] are visible to all threads, because
         * threadfence() is implied by syncthreads() */
        __syncthreads();

        /* Each thread in a warp reads cctag::any() results for one of 32 warps.
         * Redundant, but I have no better idea for spreading the result
         * to all warps. */
        // mark = threadIdx.x < HYST_H ? continuation[threadIdx.x] : false;
        mark = continuation[threadIdx.x]; // each warp reads result for all 32 warps

        /* Finally, all 32x32 threads know whether at least one of them
         * has changed a pixel.
         * If there has been any change in this round, try to spread
         * the change further.
         */
        again = cctag::any( mark );
#else
        if( threadIdx.x == 0 ) continuation[threadIdx.y] = line_changed;
        __syncthreads();
        if( threadIdx.y == 0 ) {
            mark = continuation[threadIdx.x];
            again = cctag::any(mark);
            if( threadIdx.x == 0 ) {
                continuation[0] = again;
            }
        }
        __syncthreads();
        again = continuation[0];
#endif

        /* Every threads needs to know whether any pixel was changed in
         * any round of the loop because egde_second() uses this return
         * value to write back to global memory using a different alignment. */
        if( again ) something_changed = true;

        /* this should not be necessary ... */
        debug_inner_loop_count++;
    }

    return something_changed;
}

__device__
bool edge( FrameMetaPtr& meta )
{
    bool something_changed = edge_block_loop( );
    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        if( something_changed ) {
            atomicAdd( &meta.hysteresis_block_counter(), 1 );
        }
    }
    return something_changed;
}

__global__
void edge_first( cv::cuda::PtrStepSzb img, FrameMetaPtr meta, cv::cuda::PtrStepSzb src )
{
    meta.hysteresis_block_counter() = 0;

    // const int idx  = blockIdx.x * HYST_W + threadIdx.x;
    // const int idy  = blockIdx.y * HYST_H + threadIdx.y;
    // if( outOfBounds( idx, idy, img ) ) return;
    // uint8_t val = src.ptr(idy)[idx];
    // img.ptr(idy)[idx] = val;
    cv::cuda::PtrStepSz32u input;
    input.data = reinterpret_cast<uint32_t*>(src.data);
    input.step = src.step;
    input.rows = src.rows;
    input.cols = src.cols / 4;
    load( input );

    edge( meta );

    __syncthreads();

    cv::cuda::PtrStepSz32u output;
    output.data = reinterpret_cast<uint32_t*>(img.data);
    output.step = img.step;
    output.rows = img.rows;
    output.cols = img.cols / 4;
    store( output );
}

__global__
void edge_second( cv::cuda::PtrStepSzb img, FrameMetaPtr meta )
{
    meta.hysteresis_block_counter() = 0;

    cv::cuda::PtrStepSz32u input;
    input.data = reinterpret_cast<uint32_t*>(img.data);

    input.step = img.step;
    input.rows = img.rows;
    input.cols = img.cols / 4;
    load( input );

    bool something_changed = edge( meta );

    if( cctag::any( something_changed ) ) {
        store( input );
    }
}

}; // namespace hysteresis

#ifndef NDEBUG
__global__
void verify_map_valid( cv::cuda::PtrStepSzb img, cv::cuda::PtrStepSzb ver, int w, int h )
{
    assert( img.cols == w );
    assert( img.rows == h );
    assert( ver.cols == w );
    assert( ver.rows == h );

    const int idx  = blockIdx.x * HYST_W + threadIdx.x;
    const int idy  = blockIdx.y * HYST_H + threadIdx.y;
    uint32_t x = clamp( idx, img.cols );
    uint32_t y = clamp( idy, img.rows );
    uint8_t  val = img.ptr(y)[x];
    if( val > 2 ) {
        printf("idx=%d -> x=%d, idy=%d -> y=%d, img.cols=%d img.rows=%d val=%d\n",
            idx, x, idy, y, img.cols, img.rows, val );
        assert( val <= 2 );
    }
}
#endif // NDEBUG

#ifdef USE_SEPARABLE_COMPILATION_FOR_HYST
__global__
void hyst_outer_loop_recurse( int width, int height, FrameMetaPtr meta, cv::cuda::PtrStepSzb img, cv::cuda::PtrStepSzb src, int depth )
{
    if( meta.hysteresis_block_counter() == 0 ) return;

    dim3 block;
    dim3 grid;
    block.x = HYST_W;
    block.y = HYST_H;
    grid.x  = grid_divide( width,   HYST_W * 4 );
    grid.y  = grid_divide( height,  HYST_H );

    for( int i=0; i<depth*2; i++ ) {
        hysteresis::edge_second
            <<<grid,block>>>
            ( img, meta );
    }
    hyst_outer_loop_recurse
        <<<1,1>>>
        ( width, height, meta, img, src, depth+1 );
}

__global__
void hyst_outer_loop( int width, int height, FrameMetaPtr meta, cv::cuda::PtrStepSzb img, cv::cuda::PtrStepSzb src )
{
    dim3 block;
    dim3 grid;
    block.x = HYST_W;
    block.y = HYST_H;
    grid.x  = grid_divide( width,   HYST_W * 4 );
    grid.y  = grid_divide( height,  HYST_H );

    hysteresis::edge_first
        <<<grid,block>>>
        ( img, meta, src );

    hyst_outer_loop_recurse
        <<<1,1>>>
        ( width, height, meta, img, src, 1 );

    __threadfence(); // meant to push the children's atomic meta data to CPU
}
#endif // USE_SEPARABLE_COMPILATION_FOR_HYST

__host__
void Frame::applyHyst( )
{
    assert( getWidth()  == _d_map.cols );
    assert( getHeight() == _d_map.rows );
    assert( getWidth()  == _d_hyst_edges.cols );
    assert( getHeight() == _d_hyst_edges.rows );

    dim3 block;
    dim3 grid;
    block.x = HYST_W;
    block.y = HYST_H;
    grid.x  = grid_divide( getWidth(),   HYST_W );
    grid.y  = grid_divide( getHeight(),  HYST_H );

#ifndef NDEBUG
    verify_map_valid
        <<<grid,block,0,_stream>>>
        ( _d_map, _d_hyst_edges, getWidth(), getHeight() );
#endif

#ifdef USE_SEPARABLE_COMPILATION_FOR_HYST
    hyst_outer_loop
        <<<1,1,0,_stream>>>
        ( getWidth(), getHeight(), _meta, _d_hyst_edges, _d_map );
#else // not USE_SEPARABLE_COMPILATION_FOR_HYST
    bool first_time = true;
    int  block_counter;
    do
    {
        block_counter = grid.x * grid.y;
        _meta.toDevice( Hysteresis_block_counter, block_counter, _stream );
        if( first_time ) {
            hysteresis::edge_first
                <<<grid,block,0,_stream>>>
                ( _d_hyst_edges,
                  _meta,
                  _d_map );
            first_time = false;
        } else {
            hysteresis::edge_second
                <<<grid,block,0,_stream>>>
                ( _d_hyst_edges,
                  _meta );
        }
        POP_CHK_CALL_IFSYNC;
        _meta.fromDevice( Hysteresis_block_counter, block_counter, _stream );
        POP_CUDA_SYNC( _stream );
    }
    while( block_counter > 0 );
#endif // not USE_SEPARABLE_COMPILATION_FOR_HYST
}

}; // namespace cctag


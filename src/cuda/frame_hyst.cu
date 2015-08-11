#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"
#include "assist.h"

#define ABBREVIATED_HYSTERESIS

namespace popart
{

using namespace std;

#ifdef ABBREVIATED_HYSTERESIS

namespace hysteresis
{
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

__shared__ volatile uint8_t array[HYST_H+2][HYST_W+2];

__device__
inline
uint8_t get( cv::cuda::PtrStepSzb img, const int idx, const int idy )
{
#if 1
    return img.ptr( clamp( idy, img.rows ) )[ clamp( idx, img.cols ) ];
#else
    int x = clamp( idx, img.cols );
    int y = clamp( idy, img.rows );
    assert( x >= 0 );
    assert( y >= 0 );
    assert( x < img.cols );
    assert( y < img.rows );
    uint8_t  val = img.ptr(y)[x];
    if( val > 2 ) {
        printf("idx=%d -> x=%d, idy=%d -> y=%d, img.cols=%d img.rows=%d val=%d\n",
            idx, x, idy, y, img.cols, img.rows, val );
        assert( val <= 2 );
    }
    return val;
#endif
}

__device__
void load( cv::cuda::PtrStepSzb img )
{
#if 0
    //
    // After several dead ends, this is the most trivial, most
    // brute-force approach for loading that I could find.
    //
    const int srcidx = blockIdx.x * HYST_W + threadIdx.x;
    const int srcidy = blockIdx.y * HYST_H + threadIdx.y;
    // const int dstidx = threadIdx.x;
    // const int dstidy = threadIdx.y;

    volatile uint8_t val[3][3];
    val[0][0] = get( img, srcidx-1, srcidy-1 );
    val[0][1] = get( img, srcidx  , srcidy-1 );
    val[0][2] = get( img, srcidx+1, srcidy-1 );
    val[1][0] = get( img, srcidx-1, srcidy   );
    val[1][1] = get( img, srcidx  , srcidy   );
    val[1][2] = get( img, srcidx+1, srcidy   );
    val[2][0] = get( img, srcidx-1, srcidy+1 );
    val[2][1] = get( img, srcidx  , srcidy+1 );
    val[2][2] = get( img, srcidx+1, srcidy+1 );

    assert( val[0][0] <= 2 );
    assert( val[0][1] <= 2 );
    assert( val[0][2] <= 2 );
    assert( val[1][0] <= 2 );
    assert( val[1][1] <= 2 );
    assert( val[1][2] <= 2 );
    assert( val[2][0] <= 2 );
    assert( val[2][1] <= 2 );
    assert( val[2][2] <= 2 );

    array[threadIdx.y  ][threadIdx.x  ] = val[0][0];
    array[threadIdx.y  ][threadIdx.x+2] = val[0][2];
    if( threadIdx.y >= HYST_H - 2 ) {
        array[threadIdx.y+2][threadIdx.x  ] = val[2][0];
        array[threadIdx.y+2][threadIdx.x+2] = val[2][2];
    }
    __syncthreads();
#else
    const int srcidx = blockIdx.x * HYST_W + threadIdx.x;
    const int srcidy = blockIdx.y * HYST_H + threadIdx.y;

    uint8_t val_0_0;
    val_0_0 = get( img, srcidx-1, srcidy-1 );
    array[threadIdx.y  ][threadIdx.x  ] = val_0_0;

    if( threadIdx.x >= HYST_W - 2 ) {
        uint8_t val_0_2;
        val_0_2 = get( img, srcidx+1, srcidy-1 );
        array[threadIdx.y  ][threadIdx.x+2] = val_0_2;
    }
    if( threadIdx.y >= HYST_H - 2 ) {
        uint8_t val_2_0;
        val_2_0 = get( img, srcidx-1, srcidy+1 );
        array[threadIdx.y+2][threadIdx.x  ] = val_2_0;
        if( threadIdx.x >= HYST_W - 2 ) {
            uint8_t val_2_2;
            val_2_2 = get( img, srcidx+1, srcidy+1 );
            array[threadIdx.y+2][threadIdx.x+2] = val_2_2;
        }
    }
    __syncthreads();
#endif
}

__device__
void store( cv::cuda::PtrStepSzb img )
{
    const int dstidx  = blockIdx.x * HYST_W + threadIdx.x;
    const int dstidy  = blockIdx.y * HYST_H + threadIdx.y;

    volatile uint8_t val;
    val = array[threadIdx.y+1][threadIdx.x+1];
    assert( val <= 2 );
    // if( outOfBounds( dstidx, dstidy, img ) ) return;
    if( dstidx < img.cols && dstidy < img.rows ) {
        img.ptr(dstidy)[dstidx] =  val;
    }
    // __syncthreads();
}

__device__
inline
bool update_edge_pixel( )
{
    uint8_t val[3][3];
    val[0][0] = array[threadIdx.y  ][threadIdx.x  ];
    val[0][1] = array[threadIdx.y  ][threadIdx.x+1];
    val[0][2] = array[threadIdx.y  ][threadIdx.x+2];
    val[1][0] = array[threadIdx.y+1][threadIdx.x  ];
    val[1][1] = array[threadIdx.y+1][threadIdx.x+1];
    val[1][2] = array[threadIdx.y+1][threadIdx.x+2];
    val[2][0] = array[threadIdx.y+2][threadIdx.x  ];
    val[2][1] = array[threadIdx.y+2][threadIdx.x+1];
    val[2][2] = array[threadIdx.y+2][threadIdx.x+2];

    assert( val[0][0] <= 2 );
    assert( val[0][1] <= 2 );
    assert( val[0][2] <= 2 );
    assert( val[1][0] <= 2 );
    assert( val[1][1] <= 2 );
    assert( val[1][2] <= 2 );
    assert( val[2][0] <= 2 );
    assert( val[2][1] <= 2 );
    assert( val[2][2] <= 2 );

    bool    change = false;

    if( val[1][1] == 1 ) {
        change = ( val[0][0] == 2 ||
                   val[0][1] == 2 ||
                   val[0][2] == 2 ||
                   val[1][0] == 2 ||
                   val[1][2] == 2 ||
                   val[2][0] == 2 ||
                   val[2][1] == 2 ||
                   val[2][2] == 2 );
        val[1][1] = change ? 2 : 1 ;
    }
    __syncthreads();
    array[threadIdx.y+1][threadIdx.x+1] = val[1][1];
    // __threadfence_block();

#if 0
    uint8_t test = array[threadIdx.y+1][threadIdx.x+1];
    assert( test <= 2 );
    assert( not change || test == 2 );
#endif

    return change;
}

__device__
bool edge_block_loop( )
{
    __shared__ bool continuation[HYST_H];
    bool            again = true;
    bool            nothing_changed = true;
    bool            line_changed = false;
    int ct = 0;

    while( again ) {
        assert( ct <= HYST_W*HYST_H );
        bool mark    = update_edge_pixel( );
        __threadfence();
        line_changed = __any( mark );
        if( threadIdx.x == 0 ) continuation[threadIdx.y] = line_changed;
        __syncthreads();
        mark = threadIdx.x < HYST_H ? continuation[threadIdx.x] : false;
        again = __any( mark );
        if( again ) nothing_changed = false;
        ct++;
    }

    return nothing_changed;
}

__device__
bool edge( int* block_counter )
{
    bool nothing_changed = edge_block_loop( );
    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        if( nothing_changed ) {
            atomicSub( block_counter, 1 );
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
    __syncthreads();
    return nothing_changed;
}

__global__
void edge_first( cv::cuda::PtrStepSzb img, int* block_counter, cv::cuda::PtrStepSzb src )
{
    // const int idx  = blockIdx.x * HYST_W + threadIdx.x;
    // const int idy  = blockIdx.y * HYST_H + threadIdx.y;
    // if( outOfBounds( idx, idy, img ) ) return;
    // uint8_t val = src.ptr(idy)[idx];
    // img.ptr(idy)[idx] = val;
    load( src );
    bool nothing_changed = edge( block_counter );
    store( img );
}

__global__
void edge_second( cv::cuda::PtrStepSzb img, int* block_counter )
{
    load( img );
    bool nothing_changed = edge( block_counter );
    if( not nothing_changed ) {
        store( img );
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

__host__
void Frame::applyHyst( const cctag::Parameters & params )
{
    // cerr << "Enter " << __FUNCTION__ << endl;

    dim3 block;
    dim3 grid;
    block.x = HYST_W;
    block.y = HYST_H;
    grid.x  = grid_divide( getWidth(),   HYST_W );
    grid.y  = grid_divide( getHeight(),  HYST_H );
    assert( getWidth()  == _d_map.cols );
    assert( getHeight() == _d_map.rows );
    assert( getWidth()  == _d_hyst_edges.cols );
    assert( getHeight() == _d_hyst_edges.rows );

#ifndef NDEBUG
    // cerr << "  Config: grid=" << grid << " block=" << block << endl;

    verify_map_valid
        <<<grid,block,0,_stream>>>
        ( _d_map, _d_hyst_edges, getWidth(), getHeight() );
#endif

    bool first_time = true;
    int block_counter;
#ifndef NDEBUG
    // cerr << "  Blocks remaining:";
#endif // NDEBUG
    do
    {
        block_counter = grid.x * grid.y;
        POP_CUDA_MEMCPY_TO_DEVICE_ASYNC( _d_hysteresis_block_counter,
                                         &block_counter,
                                         sizeof(int), _stream );
        if( first_time ) {
            hysteresis::edge_first
                <<<grid,block,0,_stream>>>
                ( _d_hyst_edges,
                  _d_hysteresis_block_counter,
                  _d_map );
            first_time = false;
        } else {
            hysteresis::edge_second
                <<<grid,block,0,_stream>>>
                ( _d_hyst_edges,
                  _d_hysteresis_block_counter );
        }
        POP_CHK_CALL_IFSYNC;

        POP_CUDA_MEMCPY_TO_HOST_ASYNC( &block_counter,
                                       _d_hysteresis_block_counter,
                                       sizeof(int), _stream );
        POP_CUDA_SYNC( _stream );
#ifndef NDEBUG
        // cerr << " " << block_counter;
#endif // NDEBUG
    }
    while( block_counter > 0 );
#ifndef NDEBUG
    // cerr << endl;
#endif // NDEBUG

    // cerr << "Leave " << __FUNCTION__ << endl;
}
#else // not ABBREVIATED_HYSTERESIS
#include "frame_hyst_oldcode.h"
#endif // not ABBREVIATED_HYSTERESIS

}; // namespace popart


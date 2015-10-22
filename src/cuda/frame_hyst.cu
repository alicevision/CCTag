#include <cuda_runtime.h>
#include "debug_macros.hpp"

#include "frame.h"
#include "clamp.h"
#include "assist.h"

namespace popart
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
void store( cv::cuda::PtrStepSz32u img, bool printout )
{
    const int dstidx  = blockIdx.x * HYST_W + threadIdx.x;
    const int dstidy  = blockIdx.y * HYST_H + threadIdx.y;

#if 1
    if( dstidx*sizeof(uint32_t) < img.step && dstidy < img.rows ) {
        // volatile uint32_t* shared_line = reinterpret_cast<volatile uint32_t*>(array[threadIdx.y+1]);
        volatile uint32_t* shared_line = reinterpret_cast<volatile uint32_t*>(&array[threadIdx.y+1][0]);
        uint32_t val = shared_line[threadIdx.x+1];

        img.ptr(dstidy)[dstidx] = val;
    }
#else
    union {
        uint8_t  b[4];
        uint32_t i;
    } val;

    val.b[0] = array[threadIdx.y+1][threadIdx.x*4+4];
    val.b[1] = array[threadIdx.y+1][threadIdx.x*4+5];
    val.b[2] = array[threadIdx.y+1][threadIdx.x*4+6];
    val.b[3] = array[threadIdx.y+1][threadIdx.x*4+7];
    if( printout ) {
        printf("(%d,%d)<-%d  (%d,%d)<-%d  (%d,%d)<-%d  (%d,%d)<-%d\n",
               dstidx*4+0, dstidy, val.b[0],
               dstidx*4+1, dstidy, val.b[1],
               dstidx*4+2, dstidy, val.b[2],
               dstidx*4+3, dstidy, val.b[3]);
    }
    if( dstidx*sizeof(uint32_t) < img.step && dstidy < img.rows ) {
        img.ptr(dstidy)[dstidx] = val.i;
    }
#endif
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
bool edge_block_loop( int debug_roundcount )
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
        bool line_changed = __any( mark );

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

        /* Each thread in a warp reads __any() results for one of 32 warps.
         * Redundant, but I have no better idea for spreading the result
         * to all warps. */
        // mark = threadIdx.x < HYST_H ? continuation[threadIdx.x] : false;
        mark = continuation[threadIdx.x]; // each warp reads result for all 32 warps

        /* Finally, all 32x32 threads know whether at least one of them
         * has changed a pixel.
         * If there has been any change in this round, try to spread
         * the change further.
         */
        again = __any( mark );
#else
        if( threadIdx.x == 0 ) continuation[threadIdx.y] = line_changed;
        __syncthreads();
        if( threadIdx.y == 0 ) {
            mark = continuation[threadIdx.x];
            again = __any(mark);
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
bool edge( int* block_counter, int debug_roundcount )
{
    bool something_changed = edge_block_loop( debug_roundcount );
    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        if( something_changed ) {
            if( debug_roundcount > 25 ) {
                printf("Something changed in block (%d,%d)\n", blockIdx.x, blockIdx.y );
            }
            atomicAdd( block_counter, 1 );
        }
    }
    return something_changed;
}

__global__
void edge_first( cv::cuda::PtrStepSzb img, int* block_counter, cv::cuda::PtrStepSzb src, int debug_roundcount )
{
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

    edge( block_counter, debug_roundcount );

    __syncthreads();

    cv::cuda::PtrStepSz32u output;
    output.data = reinterpret_cast<uint32_t*>(img.data);
    output.step = img.step;
    output.rows = img.rows;
    output.cols = img.cols / 4;
    store( output, false );
}

__global__
void edge_second( cv::cuda::PtrStepSzb img, int* block_counter, int debug_roundcount )
{
    cv::cuda::PtrStepSz32u input;
    input.data = reinterpret_cast<uint32_t*>(img.data);

    input.step = img.step;
    input.rows = img.rows;
    input.cols = img.cols / 4;
    load( input );

    bool something_changed = edge( block_counter, debug_roundcount );

    if( __any( something_changed ) ) {
        store( input, false );
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

#if defined(USE_SEPARABLE_COMPILATION)
__global__
void hyst_outer_loop( int width, int height, int* block_counter, cv::cuda::PtrStepSzb img, cv::cuda::PtrStepSzb src )
{
    printf( "Enter %s\n", __FUNCTION__ );

    dim3 block;
    dim3 grid;
    block.x = HYST_W;
    block.y = HYST_H;
    grid.x  = grid_divide( width,   HYST_W * 4 );
    grid.y  = grid_divide( height,  HYST_H );

    printf( "Starting (%d,%d,%d)-grid of (%d,%d,%d) threads\n",
            grid.x, grid.y, grid.z, block.x, block.y, block.z );

    bool first_time = true;
    int debug_roundcount = 0;
    do
    {
        *block_counter = 0;
        if( first_time ) {
            hysteresis::edge_first
                <<<grid,block>>>
                ( img,
                  block_counter,
                  src,
                  debug_roundcount );
            first_time = false;
        } else {
            hysteresis::edge_second
                <<<grid,block>>>
                ( img,
                  block_counter,
                  debug_roundcount );
        }
        cudaDeviceSynchronize( );
        printf( "width=%d height=%d block_counter=%d\n", width, height, *block_counter);
        assert( *block_counter <= grid.x * grid.y );
    }
    while( *block_counter > 0 && debug_roundcount++ < 30 ); // *block_counter > 0 );
    printf( "Leave %s\n", __FUNCTION__ );
}
#endif // USE_SEPARABLE_COMPILATION

__host__
void Frame::applyHyst( const cctag::Parameters & params )
{
    cerr << "Enter " << __FUNCTION__ << endl;
    assert( getWidth()  == _d_map.cols );
    assert( getHeight() == _d_map.rows );
    assert( getWidth()  == _d_hyst_edges.cols );
    assert( getHeight() == _d_hyst_edges.rows );

#ifndef NDEBUG
    dim3 block;
    dim3 grid;
    block.x = HYST_W;
    block.y = HYST_H;
    grid.x  = grid_divide( getWidth(),   HYST_W );
    grid.y  = grid_divide( getHeight(),  HYST_H );

    verify_map_valid
        <<<grid,block,0,_stream>>>
        ( _d_map, _d_hyst_edges, getWidth(), getHeight() );
#endif

#if defined(USE_SEPARABLE_COMPILATION)
    cudaEvent_t before_hyst, after_hyst;
    float ms;

    cudaEventCreate( &before_hyst );
    cudaEventCreate( &after_hyst );
    cudaEventRecord( before_hyst, _stream );
    cerr << "0" << endl;
    hyst_outer_loop
        <<<1,1,0,_stream>>>
        ( getWidth(), getHeight(), _d_hysteresis_block_counter, _d_hyst_edges, _d_map );
    cudaEventRecord( after_hyst, _stream );
    cerr << "0.1" << endl;
    cudaEventSynchronize( after_hyst );
    cerr << "0.2" << endl;
    cudaEventElapsedTime( &ms, before_hyst, after_hyst );
    cudaEventDestroy( before_hyst );
    cudaEventDestroy( after_hyst );
    std::cerr << "Hyst took " << ms << " ms" << std::endl;
#else // USE_SEPARABLE_COMPILATION
    bool first_time = true;
    int block_counter;
    do
    {
        block_counter = grid.x * grid.y;
        POP_CUDA_MEMCPY_TO_DEVICE_ASYNC( _d_hysteresis_block_counter,
                                         &block_counter,
                                         sizeof(int), _stream );
        if( first_time ) {
            cerr << "1" << endl;
            hysteresis::edge_first
                <<<grid,block,0,_stream>>>
                ( _d_hyst_edges,
                  _d_hysteresis_block_counter,
                  _d_map );
            first_time = false;
        } else {
            cerr << "2" << endl;
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
        cerr << "block_counter=" << block_counter << endl;
    }
    while( block_counter > 0 );
#endif // USE_SEPARABLE_COMPILATION
    cerr << "Leave " << __FUNCTION__ << endl;
}

}; // namespace popart


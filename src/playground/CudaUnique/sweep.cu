#include <iostream>
#include <cuda_runtime.h>
#include "assist.h"
#include "device_prop.h"

using namespace std;

const size_t NumItemPerThread = 1;
const size_t WarpSize = 32; // threadIdx.x ; threadDim.x
const size_t NumWarpsPerBlock = 32; // threadIdx.y ; threadDim.y
const size_t WarpOffset  = WarpSize * NumItemPerThread;
const size_t BlockOffset = NumWarpsPerBlock * WarpSize * NumItemPerThread;

__shared__ size_t uniqueCounter[32];

__device__
template<typename T>
T PrefixSumWarpInclusive( T threadValue, T& total )
{
    T n;
    n = __shfl_up(threadValue,  1); if( threadIdx.x >=  1 ) threadValue += n;
    n = __shfl_up(threadValue,  2); if( threadIdx.x >=  2 ) threadValue += n;
    n = __shfl_up(threadValue,  4); if( threadIdx.x >=  4 ) threadValue += n;
    n = __shfl_up(threadValue,  8); if( threadIdx.x >=  8 ) threadValue += n;
    n = __shfl_up(threadValue, 16); if( threadIdx.x >= 16 ) threadValue += n;
    total        = __shfl(    threadValue, 31 );
    return threadValue;
}

__device__
template<typename T>
T PrefixSumWarpExclusive( T threadValue, T& total )
{
    T n;
    n = __shfl_up(threadValue,  1); if( threadIdx.x >=  1 ) threadValue += n;
    n = __shfl_up(threadValue,  2); if( threadIdx.x >=  2 ) threadValue += n;
    n = __shfl_up(threadValue,  4); if( threadIdx.x >=  4 ) threadValue += n;
    n = __shfl_up(threadValue,  8); if( threadIdx.x >=  8 ) threadValue += n;
    n = __shfl_up(threadValue, 16); if( threadIdx.x >= 16 ) threadValue += n;

    total = __shfl(    threadValue, 31 );
    n  = __shfl_up( threadValue, 1 );
    threadValue = threadIdx.x == 0 ? 0 : n;
    return threadValue;
}

__device__
size_t PrefixSumBlockExclusive( size_t threadValue, size_t& blockTotal )
{
    size_t offset;
    size_t warpTotal;
    size_t n;

    // pre: every threadValue is either 1 (keep) or 0 (drop)
    n = PrefixSumWarpExclusive( threadValue, warpTotal );
    // post: every thread knows its write offset relative to warp base
    threadValue = n;

    if( threadIdx.x == 0 ) {
        uniqueCounter[threadIdx.y] = warpTotal;
        // post: uniqueCounter contains the warps' number of kept items
    }
    __syncthreads();

    if( threadIdx.y == 0 ) {
        offset = uniqueCounter[threadIdx.x];
        n = PrefixSumWarpExclusive( offset, blockTotal );
        // blockTotal is only valid for threadIdx.y == 0
        uniqueCounter[threadIdx.x] = n;
    }
    __syncthreads();

    offset = uniqueCounter[threadIdx.y];

    threadValue += offset;
    // post: every thread knows its write offset relative to block base

    return threadValue;
}

__global__
void SweepEqualityBlock( const int32_t* in_array,
                         int16_t*       out_offset_array,
                         int32_t*       out_block_total,
                         const size_t   num_in )
{
    int baseOffset = blockIdx.x * BlockOffset + threadIdx.y * WarpOffset + threadIdx.x;
    int leftOffset = baseOffset - 1 ;
    int ridx = min( baseOffset, (int)(num_in-1) );
    int lidx = max( 0, min( leftOffset, (int)(num_in-1) ) );
    int32_t l = in_array[lidx];
    int32_t r = in_array[ridx];
    size_t  isUniqueElem = ( ridx == 0 ) ? 1 : ( l == r ) ? 0 : 1;
    size_t  blockTotal;
    int16_t writeOffset = (int16_t)PrefixSumBlockExclusive( isUniqueElem, blockTotal );
    if( baseOffset < num_in ) {
        out_offset_array[baseOffset] = isUniqueElem ? writeOffset : -1;
    }
    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        out_block_total[ blockIdx.x ] = blockTotal;
    }
}

__global__
void SumEqualityBlock( const int      items_per_thread,
                       const int32_t* in_block_total,
                       int32_t*       out_block_sum,
                       const int      num_items )
{
    size_t offset = ( threadIdx.y * 32 + threadIdx.x ) * items_per_thread;
    int32_t counter = 0;
    for( int i=0; i<items_per_thread; i++ ) {
        counter += (offset+i < num_items ) ? in_block_total[offset+i] : 0;
    }
    size_t total;
    size_t exclusiveSum = PrefixSumBlockExclusive( counter, total );
    counter = 0;
    for( int i=0; i<items_per_thread; i++ ) {
        if( offset+i >= num_items ) return;
        int32_t counterIncrease = in_block_total[offset+i];
        out_block_sum[offset+i] = exclusiveSum + counter;
        counter += counterIncrease;
    }
}

/* must have the same block structure as SweepEqualityBlock */
__global__
void WriteUniqueValues( const int32_t* in_array,
                        const int32_t* in_block_sum,
                        const int16_t* in_offset_array,
                        int32_t*       out_array,
                        const size_t   num_in )
{
    size_t baseOffset = blockIdx.x * BlockOffset + threadIdx.y * WarpOffset + threadIdx.x;
    if( baseOffset >= num_in ) return;
    int16_t writeIndex = in_offset_array[baseOffset];
    if( writeIndex == -1 ) return;
    writeIndex += in_block_sum[blockIdx.x];
    out_array[writeIndex] = in_array[baseOffset];
}

#if 0
__global__
void print_d_intermediate_offset_array( int16_t* a, size_t num )
{
    printf( "intermediate offset array:\n" );
    for( int i=0; i<num; i++ ) {
        printf("%d ", (int)a[i] );
    }
    printf("\n");
}

__global__
void print_d_intermediate_block_total( int32_t* a, size_t num )
{
    printf( "intermediate block total:\n" );
    for( int i=0; i<num; i++ ) {
        printf("%d ", a[i] );
    }
    printf("\n");
}
#endif


void UniqueArray( int32_t*  h_ptr,
                  size_t    num_in,
                  int32_t*  d_ptr_in,
                  int32_t*  d_ptr_out,
                  int16_t*  d_intermediate_offset_array,
                  int32_t*  d_intermediate_block_total,
                  int32_t*  d_intermediate_block_sum,
                  int32_t* d_intermediate_3 )
{
    cout << "Enter " << __FUNCTION__ << endl;

    int16_t*  h_ptr_intermediate_1;
    int32_t*  h_ptr_intermediate_2;
    int32_t*  h_ptr_intermediate_3;
    int32_t*  h_ptr_intermediate_4;
    cudaMallocHost( &h_ptr_intermediate_1, num_in*sizeof(int16_t) );
    cudaMallocHost( &h_ptr_intermediate_2, num_in*sizeof(int32_t) );
    cudaMallocHost( &h_ptr_intermediate_3, num_in*sizeof(int32_t) );
    cudaMallocHost( &h_ptr_intermediate_4, sizeof(int32_t) );

    cout << "Num init values: " << num_in << endl;

    cudaError_t err;

    err = cudaMemcpy( d_ptr_in, h_ptr, num_in*sizeof(int32_t), cudaMemcpyHostToDevice );
    if( err != cudaSuccess ) {
        cout << "Error in line " << __LINE__ << ": " << cudaGetErrorString(err) << endl;
    }

    int  block_count = grid_divide( num_in, 32*32 );

    cout << "Required 32x32 (" << 32*32 << ") blocks: " << block_count << endl;

    dim3 block( 32, 32 );
    dim3 grid ( block_count );

    SweepEqualityBlock
        <<<grid,block>>>
        ( d_ptr_in,
          d_intermediate_offset_array,
          d_intermediate_block_total,
          num_in );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        cout << "Error in line " << __LINE__ << ": " << cudaGetErrorString(err) << endl;
    }

    cudaMemcpy( h_ptr_intermediate_1,
                d_intermediate_offset_array,
                num_in*sizeof(int16_t),
                cudaMemcpyDeviceToHost );

    cout << "Intermediate offset array:" << endl;
    for( int i=0; i<num_in; i++ ) {
        cout << h_ptr_intermediate_1[i] << " ";
        if( i%16==15 ) cout << endl;
    }
    cout << endl;

    cudaMemcpy( h_ptr_intermediate_2,
                d_intermediate_block_total,
                block_count*sizeof(int32_t),
                cudaMemcpyDeviceToHost );

    cout << "Intermediate block total:" << endl;
    for( int i=0; i<block_count; i++ ) {
        cout << h_ptr_intermediate_2[i] << " ";
        if( i%16==15 ) cout << endl;
    }
    cout << endl;

    SumEqualityBlock
        <<<1,block>>>
        ( grid_divide( num_in, 32*32 ),
          d_intermediate_block_total,
          d_intermediate_block_sum,
          num_in );

    cout << "Intermediate block sum:" << endl;
    for( int i=0; i<block_count; i++ ) {
        cout << h_ptr_intermediate_2[i] << " ";
        if( i%16==15 ) cout << endl;
    }
    cout << endl;

    WriteUniqueValues
        <<<grid,block>>>
        ( d_ptr_in,
          d_intermediate_block_sum,
          d_intermediate_offset_array,
          d_ptr_out,
          num_in );

    cudaMemcpy( h_ptr, d_ptr_out, num_in*sizeof(int32_t), cudaMemcpyDeviceToHost );

    cout << "Leave " << __FUNCTION__ << endl;
}

int main( )
{
    device_prop_t dev;
    dev.print();

    const size_t num = 2000;
    int32_t*  h_ptr;
    int32_t*  d_ptr_in;
    int32_t*  d_ptr_out;
    int16_t*  d_ptr_intermediate_1;
    int32_t*  d_ptr_intermediate_2;
    int32_t*  d_ptr_intermediate_3;
    int32_t*  d_ptr_intermediate_4;
    cudaMallocHost( &h_ptr, num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_in,  num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_out, num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_intermediate_1, num*sizeof(int16_t) );
    cudaMalloc( &d_ptr_intermediate_2, num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_intermediate_3, num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_intermediate_4, sizeof(int32_t) );
    for( int i=0; i<num; i++ ) {
        h_ptr[i] = random() % 10;
    }
    cout << "Input array:" << endl;
    for( int i=0; i<num; i++ ) {
        cout << h_ptr[i] << " ";
        if( i%16==15 ) cout << endl;
    }
    cout << endl;
    UniqueArray( h_ptr,
                 num,
                 d_ptr_in,
                 d_ptr_out,
                 d_ptr_intermediate_1,
                 d_ptr_intermediate_2,
                 d_ptr_intermediate_3,
                 d_ptr_intermediate_4 );
    cout << "Output array:" << endl;
    for( int i=0; i<num; i++ ) {
        cout << h_ptr[i] << " ";
        if( i%16==15 ) cout << endl;
    }
    cout << endl;
}


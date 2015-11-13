#include <iostream>
#include <cuda_runtime.h>
#include "assist.h"
#include "device_prop.h"
#include "d_warp_prefixsum.h"

using namespace std;

const size_t NumItemPerThread = 1;
const size_t WarpSize = 32; // threadIdx.x ; threadDim.x
const size_t NumWarpsPerBlock = 32; // threadIdx.y ; threadDim.y
const size_t WarpOffset  = WarpSize * NumItemPerThread;
const size_t BlockOffset = NumWarpsPerBlock * WarpSize * NumItemPerThread;

namespace popart {

__shared__ size_t uniqueCounter[32];

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
                       const int      in_block_items,
                       int32_t*       out_block_prefixsum,
                       size_t*        out_block_overallsum )
{
    size_t offset = ( threadIdx.y * 32 + threadIdx.x ) * items_per_thread;
    int32_t counter = 0;
    for( int i=0; i<items_per_thread; i++ ) {
        counter += (offset+i < in_block_items ) ? in_block_total[offset+i] : 0;
    }
    size_t total;
    size_t exclusiveSum = PrefixSumBlockExclusive( counter, total );
    counter = 0;
    for( int i=0; i<items_per_thread; i++ ) {
        if( offset+i >= in_block_items ) return;
        int32_t counterIncrease = in_block_total[offset+i];
        out_block_prefixsum[offset+i] = exclusiveSum + counter;
        counter += counterIncrease;
    }
    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        *out_block_overallsum = total;
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


void UniqueArray( int32_t* h_ptr,
                  size_t&  h_num_out,
                  size_t   num_in,
                  int32_t* d_ptr_in,
                  int32_t* d_ptr_out,
                  int16_t* d_intermediate_offset_array,
                  int32_t* d_intermediate_block_total,
                  int32_t* d_intermediate_block_sum,
                  size_t*  d_intermediate_num_output_items )
{
    cout << "Enter " << __FUNCTION__ << endl;

    cudaError_t err;

    err = cudaMemcpy( d_ptr_in, h_ptr, num_in*sizeof(int32_t), cudaMemcpyHostToDevice );
    if( err != cudaSuccess ) {
        cout << "Error in line " << __LINE__ << ": " << cudaGetErrorString(err) << endl;
    }

    int  block_count = grid_divide( num_in, 32*32 );

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

    SumEqualityBlock
        <<<1,block>>>
        ( grid_divide( block_count, 32*32 ),
          d_intermediate_block_total,
          block_count,
          d_intermediate_block_sum,
          d_intermediate_num_output_items );

    cudaMemcpy( &h_num_out, d_intermediate_num_output_items, sizeof(size_t), cudaMemcpyDeviceToHost );

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

} // namespace popart

using namespace popart;

int main( )
{
    device_prop_t dev;
    dev.print();

    const size_t num = 3000;
    size_t    h_num_out;
    int32_t*  h_ptr;
    int32_t*  d_ptr_in;
    int32_t*  d_ptr_out;
    int16_t*  d_ptr_intermediate_1;
    int32_t*  d_ptr_intermediate_2;
    int32_t*  d_ptr_intermediate_3;
    size_t*   d_ptr_intermediate_4;
    cudaMallocHost( &h_ptr, num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_in,  num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_out, num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_intermediate_1, num*sizeof(int16_t) );
    cudaMalloc( &d_ptr_intermediate_2, num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_intermediate_3, num*sizeof(int32_t) );
    cudaMalloc( &d_ptr_intermediate_4, sizeof(size_t) );
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
                 h_num_out,
                 num,
                 d_ptr_in,
                 d_ptr_out,
                 d_ptr_intermediate_1,
                 d_ptr_intermediate_2,
                 d_ptr_intermediate_3,
                 d_ptr_intermediate_4 );
    cout << "Unique output item: " << h_num_out << endl;
    cout << "Output array:" << endl;
    for( int i=0; i<num; i++ ) {
        cout << h_ptr[i] << " ";
        if( i%16==15 ) cout << endl;
    }
    cout << endl;
}


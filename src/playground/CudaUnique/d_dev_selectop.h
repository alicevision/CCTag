#pragma once

#include <cuda_runtime.h>
#include "d_block_prefixsum.h"

namespace popart {

template<typename OpTest, typename IndexCtType>
__global__
void SweepBlockOp( const OpTest& in,
                   IndexCtType*  out_offset_array,
                   IndexCtType*  out_block_total )
{
    int baseOffset = blockIdx.x * BlockOffset + threadIdx.y * WarpOffset + threadIdx.x;
    // size_t  isUniqueElem = in.operator()(baseOffset) ? 1 : 0;
    IndexCtType isUniqueElem = in(baseOffset) ? 1 : 0;
    IndexCtType blockTotal;
    IndexCtType writeOffset = PrefixSumBlockExclusive( isUniqueElem, blockTotal );
    if( baseOffset < num_in ) {
        out_offset_array[baseOffset] = isUniqueElem ? writeOffset : -1;
    }
    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        out_block_total[ blockIdx.x ] = blockTotal;
    }
}

template<typename IndexCtType>
__global__
void SumBlockOp( const int                       items_per_thread,
                 const IndexCtType* __restrict__ in_block_total,
                 const int                       in_block_items,
                 IndexCtType* __restrict__       out_block_prefixsum,
                 IndexCtType* __restrict__       out_block_overallsum )
{
    size_t      offset = ( threadIdx.y * 32 + threadIdx.x ) * items_per_thread;
    IndexCtType counter = 0;
    for( int i=0; i<items_per_thread; i++ ) {
        counter += (offset+i < in_block_items ) ? in_block_total[offset+i] : 0;
    }
    // post: each thread holds the sum of items_per_thread consecutive counts
    //       no need for __syncthreads: __shfl_up in PrefixSumBlockExclusive
    //       is sync'ingt

    IndexCtType total;
    IndexCtType exclusiveSum = PrefixSumBlockExclusive( counter, total );
    // post: each thread knows the base count for it's block of counts
    //       All threads with threadIdx.y == 0 know the total sum of all
    //       counts. total is undefined for all other threads.

    counter = 0;
    for( int i=0; i<items_per_thread; i++ ) {
        if( offset+i >= in_block_items ) return;
        IndexCtType counterIncrease = in_block_total[offset+i];
        out_block_prefixsum[offset+i] = exclusiveSum + counter;
        counter += counterIncrease;
    }
    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        *out_block_overallsum = total;
    }
}

/* must have the same block structure as SweepBlockOp */
template<typename T, typename IndexCtType>
__global__
void WriteSelectedOp( const T* __restrict__           in_array,
                      const IndexCtType* __restrict__ in_block_sum,
                      const IndexCtType* __restrict__ in_offset_array,
                      T* __restrict__                 out_array,
                      const size_t                    num_in )
{
    size_t baseOffset = blockIdx.x * BlockOffset + threadIdx.y * WarpOffset + threadIdx.x;
    if( baseOffset >= num_in ) return;

    IndexCtType writeIndex = in_offset_array[baseOffset];
    if( writeIndex == -1 ) return;
    writeIndex += in_block_sum[blockIdx.x];

    const T movedValue = in_array[baseOffset];
    out_array[writeIndex] = movedValue;
}

} //namespace popart


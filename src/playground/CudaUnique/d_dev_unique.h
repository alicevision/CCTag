#pragma once

#include <cuda_runtime.h>
#include "d_block_prefixsum.h"
#include "d_dev_selectop.h"

namespace popart {

template<typename T>
struct OpTestUnique
{
    const T* array; // on device
    size_t   size;

    __device__
    OpTestUnique( const T* in_array, size_t sz )
        : array( in_array )
        , size( sz )
    { }

    __device__
    bool operator()( int right_idx ) const {
        const int left_idx = right_idx - 1;
        const int ridx = min( right_idx, (int)(size-1) );
        const int lidx = max( 0, min( left_idx, (int)(size-1) ) );
        const T l = array[lidx];
        const T r = array[ridx];
        const bool isUniqueElem = ( ridx == 0 ) ? true : ( l == r ) ? false : true;
        return isUniqueElem;
    }
};

template<typename T, typename IndexCtType>
class Unique
{
    OpTestUnique<T> _optest;

    size_t requiredIntermediateByteMemory( size_t num_in ) const {
        const size_t sz_offset_array = num_in * sizeof( IndexCtType );
        const size_t block_count     = grid_divide( num_in, 32*32 );
        const size_t sz_block_total  = block_count * sizeof( IndexCtType );
        const size_t sz_block_sum    = block_count * sizeof( IndexCtType );
        const size_t sz_out_item_ct  = sizeof( IndexCtType );
    }
};

template<typename T, typename IndexCtType>
void UniqueArray( T* h_ptr,
                  size_t&  h_num_out,
                  size_t   num_in,
                  T* __restrict__ d_ptr_in,
                  T* __restrict__ d_ptr_out,
                  IndexCtType* d_intermediate_offset_array,
                  IndexCtType* d_intermediate_block_total,
                  IndexCtType* d_intermediate_block_sum,
                  IndexCtType* d_intermediate_num_output_items )
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

    OpTestUnique<int32_t> op( d_ptr_in, num_in );

    SweepBlockOp<OpTestUnique,int32_t>
        <<<grid,block>>>
        ( op,
          d_intermediate_offset_array,
          d_intermediate_block_total );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        cout << "Error in line " << __LINE__ << ": " << cudaGetErrorString(err) << endl;
    }

    SumBlockOp<int32_t>
        <<<1,block>>>
        ( grid_divide( block_count, 32*32 ),
          d_intermediate_block_total,
          block_count,
          d_intermediate_block_sum,
          d_intermediate_num_output_items );

    cudaMemcpy( &h_num_out, d_intermediate_num_output_items, sizeof(size_t), cudaMemcpyDeviceToHost );

    WriteSelectedOp<int32_t,int32_t>
        <<<grid,block>>>
        ( d_ptr_in,
          d_intermediate_block_sum,
          d_intermediate_offset_array,
          d_ptr_out,
          num_in );

    cudaMemcpy( h_ptr, d_ptr_out, num_in*sizeof(int32_t), cudaMemcpyDeviceToHost );

    cout << "Leave " << __FUNCTION__ << endl;
}

} //namespace popart


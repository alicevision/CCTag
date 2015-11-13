#pragma once

#include <cuda_runtime.h>
#include "d_block_prefixsum.h"
#include "d_dev_selectop.h"

namespace popart {

// struct OpTest { bool operator( int idx ) const; };

template<typename T>
struct OpTestUnique : public OpTest
{
    const T* array; // on device
    size_t   size;

    __device__
    OpTestUnique( const T* in_array, size_t sz )
        : array( in_array )
        , size( sz )
    { }

    bool operator( int right_idx ) const {
        const int left_idx = right_idx - 1;
        const int ridx = min( right_idx, (int)(size-1) );
        const int lidx = max( 0, min( left_idx, (int)(size-1) ) );
        const T l = array[lidx];
        const T r = array[ridx];
        const bool isUniqueElem = ( ridx == 0 ) ? true : ( l == r ) ? false : true;
        return isUniqueElem;
    }
};

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


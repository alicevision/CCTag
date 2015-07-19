// #include <iostream>
// #include <limits>
#include <cuda_runtime.h>
// #include <stdio.h>
#include "debug_macros.hpp"

#include "frame.h"

namespace popart
{

using namespace std;

__host__
void Frame::allocDevGaussianPlane( const cctag::Parameters& params )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    void* ptr;
    const size_t w = getWidth();
    const size_t h = getHeight();
    size_t p;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    assert( p % _d_smooth.elemSize() == 0 );
    _d_smooth.data = (float*)ptr;
    _d_smooth.step = p;
    _d_smooth.cols = w;
    _d_smooth.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(int16_t), h );
    assert( p % _d_dx.elemSize() == 0 );
    _d_dx.data = (int16_t*)ptr;
    _d_dx.step = p;
    _d_dx.cols = w;
    _d_dx.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(int16_t), h );
    assert( p % _d_dy.elemSize() == 0 );
    _d_dy.data = (int16_t*)ptr;
    _d_dy.step = p;
    _d_dy.cols = w;
    _d_dy.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(float), h );
    assert( p % _d_intermediate.elemSize() == 0 );
    _d_intermediate.data = (float*)ptr;
    _d_intermediate.step = p;
    _d_intermediate.cols = w;
    _d_intermediate.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(uint32_t), h );
    assert( p % _d_mag.elemSize() == 0 );
    _d_mag.data = (uint32_t*)ptr;
    _d_mag.step = p;
    _d_mag.cols = w;
    _d_mag.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(unsigned char), h );
    assert( p % _d_map.elemSize() == 0 );
    _d_map.data = (unsigned char*)ptr;
    _d_map.step = p;
    _d_map.cols = w;
    _d_map.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(unsigned char), h );
    assert( p % _d_hyst_edges.elemSize() == 0 );
    _d_hyst_edges.data = (unsigned char*)ptr;
    _d_hyst_edges.step = p;
    _d_hyst_edges.cols = w;
    _d_hyst_edges.rows = h;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(unsigned char), h );
    assert( p % _d_edges.elemSize() == 0 );
    _d_edges.data = (unsigned char*)ptr;
    _d_edges.step = p;
    _d_edges.cols = w;
    _d_edges.rows = h;

    POP_CUDA_MALLOC( &ptr, sizeof(int) );
    _d_hysteresis_block_counter        = (int*)ptr;

    POP_CUDA_MALLOC( &ptr, sizeof(int) );
    _d_connect_component_block_counter = (int*)ptr;

    _vote.alloc( params, w, h );

    POP_CUDA_MEMSET_ASYNC( _d_smooth.data,
                           0,
                           _d_smooth.step * _d_smooth.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_dx.data,
                           0,
                           _d_dx.step * _d_dx.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_dy.data,
                           0,
                           _d_dy.step * _d_dy.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_intermediate.data,
                           0,
                           _d_intermediate.step * _d_intermediate.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_mag.data,
                           0,
                           _d_mag.step * _d_mag.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_map.data,
                           0,
                           _d_map.step * _d_map.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_edges.data,
                           0,
                           _d_edges.step * _d_edges.rows,
                           _stream );

    _vote.init( params, _stream );

    // POP_CUDA_MEMSET_ASYNC( _d_next_edge_after.data,
    //                        0,
    //                        _d_next_edge_after.step * _d_next_edge_after.rows,
    //                        _stream );

    // POP_CUDA_MEMSET_ASYNC( _d_next_edge_befor.data,
    //                        0,
    //                        _d_next_edge_befor.step * _d_next_edge_befor.rows,
    //                        _stream );

    cerr << "Leave " << __FUNCTION__ << endl;
}

void Voting::alloc( const cctag::Parameters& params, size_t w, size_t h )
{
    void*  ptr;
    size_t p;

    POP_CUDA_MALLOC( &ptr, params._maxEdges*sizeof(int2) );
    _all_edgecoords.dev.ptr = (int2*)ptr;

    POP_CUDA_MALLOC( &ptr, sizeof(int) );
    _all_edgecoords.dev.size = (int*)ptr;

    POP_CUDA_MALLOC( &ptr, params._maxEdges*sizeof(TriplePoint) );
    _chained_edgecoords.dev.ptr = (TriplePoint*)ptr;

    POP_CUDA_MALLOC( &ptr, sizeof(int) );
    _chained_edgecoords.dev.size = (int*)ptr;

    POP_CUDA_MALLOC( &ptr, params._maxEdges*sizeof(int) );
    _edge_indices.dev.ptr = (int*)ptr;

    POP_CUDA_MALLOC( &ptr, sizeof(int) );
    _edge_indices.dev.size = (int*)ptr;

    POP_CUDA_MALLOC( &ptr, params._maxEdges*sizeof(int) );
    _edge_indices_2.dev.ptr = (int*)ptr;

    POP_CUDA_MALLOC( &ptr, sizeof(int) );
    _edge_indices_2.dev.size = (int*)ptr;

    POP_CUDA_MALLOC_PITCH( &ptr, &p, w*sizeof(int32_t), h );
    assert( p % _d_edgepoint_index_table.elemSize() == 0 );
    _d_edgepoint_index_table.data = (int32_t*)ptr;
    _d_edgepoint_index_table.step = p;
    _d_edgepoint_index_table.cols = w;
    _d_edgepoint_index_table.rows = h;
}

void Voting::init( const cctag::Parameters& params, cudaStream_t stream )
{
    POP_CUDA_MEMSET_ASYNC( _all_edgecoords.dev.ptr,
                           0,
                           params._maxEdges*sizeof(int2),
                           stream );

    POP_CUDA_MEMSET_ASYNC( _chained_edgecoords.dev.ptr,
                           0,
                           params._maxEdges*sizeof(TriplePoint),
                           stream );

    POP_CUDA_MEMSET_ASYNC( _edge_indices.dev.ptr,
                           0,
                           params._maxEdges*sizeof(int),
                           stream );

    POP_CUDA_MEMSET_ASYNC( _edge_indices_2.dev.ptr,
                           0,
                           params._maxEdges*sizeof(int),
                           stream );

    POP_CUDA_MEMSET_ASYNC( _d_edgepoint_index_table.data,
                           0,
                           _d_edgepoint_index_table.step * _d_edgepoint_index_table.rows,
                           stream );
}

void Voting::release( )
{
    POP_CUDA_FREE( _all_edgecoords.dev.ptr );
    POP_CUDA_FREE( _all_edgecoords.dev.size );
    POP_CUDA_FREE( _chained_edgecoords.dev.ptr );
    POP_CUDA_FREE( _chained_edgecoords.dev.size );
    POP_CUDA_FREE( _edge_indices.dev.ptr );
    POP_CUDA_FREE( _edge_indices.dev.size );
    POP_CUDA_FREE( _edge_indices_2.dev.ptr );
    POP_CUDA_FREE( _edge_indices_2.dev.size );
    POP_CUDA_FREE( _d_edgepoint_index_table.data );
}

}; // namespace popart


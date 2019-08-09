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

namespace cctag
{

using namespace std;

__host__
void Frame::allocRequiredMem( const cctag::Parameters& params )
{
    _meta.toDevice( Ring_counter_max, EDGE_LINKING_MAX_ARCS, _stream );

    void* ptr;
    const size_t w = getWidth();
    const size_t h = getHeight();
    size_t p;

    _d_smooth    .allocate( h, w );
    _d_dx        .allocate( h, w );
    _d_dy        .allocate( h, w );
    _d_mag       .allocate( h, w );
    _d_map       .allocate( h, w );
    _d_hyst_edges.allocate( h, w );
    _d_edges     .allocate( h, w );

    _h_plane     .allocate( h, w );
    _h_dx        .allocate( h, w );
    _h_dy        .allocate( h, w );
    _h_mag       .allocate( h, w );
    _h_edges     .allocate( h, w );

    _d_ring_output.allocate( EDGE_LINKING_MAX_ARCS,             // height
                             EDGE_LINKING_MAX_EDGE_LENGTH );    // width

    _d_intermediate.allocate( h, w );
    _h_intermediate.allocate( _d_intermediate.rows,
                              _d_intermediate.step / sizeof(float) );

#ifdef DEBUG_WRITE_MAP_AS_PGM
    POP_CUDA_MALLOC_HOST( &ptr, w * h * sizeof(unsigned char) );
    _h_debug_map = (unsigned char*)ptr;
#endif // DEBUG_WRITE_MAP_AS_PGM

    _all_edgecoords     .alloc( EDGE_POINT_MAX, EdgeListBoth );
    _voters             .alloc( EDGE_POINT_MAX, EdgeListBoth );
    _v_chosen_idx       .alloc( EDGE_POINT_MAX, EdgeListBoth );
    _inner_points       .alloc( EDGE_POINT_MAX, EdgeListBoth );
    _interm_inner_points.alloc( EDGE_POINT_MAX, EdgeListDevOnly );

    POP_CUDA_MALLOC( &ptr, EDGE_POINT_MAX * sizeof(float) );
    _v_chosen_flow_length = (float*)ptr;

    _vote._d_edgepoint_index_table.allocate( h, w );
}

__host__
void Frame::initRequiredMem( )
{
    POP_CUDA_MEMSET_ASYNC( _d_smooth.data,
                           0,
                           _d_smooth.step * _d_smooth.rows,
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

    POP_CUDA_MEMSET_ASYNC( _d_dx.data,
                           0,
                           _d_dx.step * _d_dx.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_dy.data,
                           0,
                           _d_dy.step * _d_dy.rows,
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _d_edges.data,
                           0,
                           _d_edges.step * _d_edges.rows,
                           _stream );

    _all_edgecoords.init( _stream );
    _voters        .init( _stream );
    _v_chosen_idx  .init( _stream );
    _inner_points  .init( _stream );
    _interm_inner_points.init( _stream );

    POP_CUDA_MEMSET_ASYNC( _v_chosen_flow_length,
                           0,
                           EDGE_POINT_MAX * sizeof(float),
                           _stream );

    POP_CUDA_MEMSET_ASYNC( _vote._d_edgepoint_index_table.data,
                           0,
                           _vote._d_edgepoint_index_table.step * _vote._d_edgepoint_index_table.rows,
                           _stream );
}

void Frame::releaseRequiredMem( )
{
    POP_CUDA_FREE( _d_plane.data );

    // allocated in allocRequiredMem
    _d_smooth    .release();
    _d_dx        .release();
    _d_dy        .release();
    _d_mag       .release();
    _d_map       .release();
    _d_hyst_edges.release();
    _d_edges     .release();

    _h_plane     .release();
    _h_dx        .release();
    _h_dy        .release();
    _h_mag       .release();
    _h_edges     .release();

    _d_ring_output.release();

    _d_intermediate.release();
    _h_intermediate.release();

#ifdef DEBUG_WRITE_MAP_AS_PGM
    POP_CUDA_FREE_HOST( _h_debug_map );
#endif // DEBUG_WRITE_MAP_AS_PGM

    _all_edgecoords.release();
    _voters        .release();
    _v_chosen_idx  .release();
    _inner_points  .release();
    _interm_inner_points.release();

    POP_CUDA_FREE( _v_chosen_flow_length );

    _vote._d_edgepoint_index_table.release();
}

}; // namespace cctag


/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>
#include "onoff.h"

#define FRAME_META_MAX_PIPES	MAX_PIPES
#define FRAME_META_MAX_LEVELS	8

namespace cctag {

enum FrameMetaEnum
{
    Hysteresis_block_counter,         // int
    Connect_component_block_counter,  // int
    Ring_counter,                     // int
    Ring_counter_max,                 // int
    Identification_result,            // float
    Identification_resct,             // int
#ifndef NDEBUG
    Num_edges_thinned,
#endif // NDEBUG
    List_size_all_edgecoords,
    List_size_voters,
    List_size_chosen_idx,
    List_size_inner_points,
    List_size_interm_inner_points,
    Swap_buffers_after_sort,
    End_indicator
};

struct FrameMetaPtr
{
    void* _d_symbol_ptr;
    int   _pipeId;
    int   _frameId;

    /* Create this FrameMeta pointer. Only the host can do it.
     * But it is passed to the device to use all those accessor
     * functions.
     */
    __host__
    FrameMetaPtr( int pipeId, int frameId );

    /* Copy the symbol e from this pipe/frame from host to device
     */
    __host__
    void toDevice( FrameMetaEnum e, int val, cudaStream_t stream );
    __host__
    void toDevice( FrameMetaEnum e, float val, cudaStream_t stream );

    /* Copy the symbol e from a device location from device to device
     */
    __host__
    void toDevice_D2S( FrameMetaEnum e, int* d_val, cudaStream_t stream );
    __host__
    void toDevice_D2S( FrameMetaEnum e, float* d_val, cudaStream_t stream );

    /* Copy the symbol e from this pipe/frame from device to host
     */
    __host__
    void fromDevice( FrameMetaEnum e, int& val, cudaStream_t stream );
    __host__
    void fromDevice( FrameMetaEnum e, float& val, cudaStream_t stream );

#define OFFSET_GETTER_HEADER( type, name ) \
    __device__ type& name(); \
    __device__ const type& name() const;

    OFFSET_GETTER_HEADER( int, hysteresis_block_counter )
    OFFSET_GETTER_HEADER( int, connect_component_block_counter )
    OFFSET_GETTER_HEADER( int, ring_counter )
    OFFSET_GETTER_HEADER( int, ring_counter_max )
    OFFSET_GETTER_HEADER( float, identification_result )
    OFFSET_GETTER_HEADER( int, identification_resct )
#ifndef NDEBUG
    OFFSET_GETTER_HEADER( int, num_edges_thinned )
#endif // NDEBUG
    OFFSET_GETTER_HEADER( int, list_size_all_edgecoords )
    OFFSET_GETTER_HEADER( int, list_size_voters )
    OFFSET_GETTER_HEADER( int, list_size_chosen_idx )
    OFFSET_GETTER_HEADER( int, list_size_inner_points )
    OFFSET_GETTER_HEADER( int, list_size_interm_inner_points )
    OFFSET_GETTER_HEADER( int, swap_buffers_after_sort )

private:
    // These default functions are actually needed for automatic
    // host-device copying:
    // FrameMetaPtr( );
    // FrameMetaPtr( const FrameMetaPtr& );
    // FrameMetaPtr& operator=( const FrameMetaPtr& );
};

} // namespace cctag


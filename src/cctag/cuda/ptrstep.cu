/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "cctag/cuda/ptrstep.h"
#include "cctag/cuda/debug_macros.hpp"

namespace cctag {

/*************************************************************
 * allocation functions
 *************************************************************/
void* allocate_hst( size_t h, size_t w_bytes, size_t& w_pitch )
{
    void* ptr;
    POP_CUDA_MALLOC_HOST( &ptr, h*w_bytes );
    w_pitch = w_bytes;
    return ptr;
}

void* allocate_dev( size_t h, size_t w_bytes, size_t& w_pitch )
{
    void*  ptr;
    POP_CUDA_MALLOC_PITCH( &ptr, &w_pitch, w_bytes, h );
    return ptr;
}

void release_hst( void* ptr )
{
    POP_CUDA_FREE_HOST( ptr );
}

void release_dev( void* ptr )
{
    POP_CUDA_FREE( ptr );
}

/*************************************************************
 * HstPlane2DbClone
 *************************************************************/
HstPlane2DbClone::HstPlane2DbClone( const HstPlane2Db& orig )
    : e ( orig )
{
    e.data = new uint8_t[ orig.rows * orig.step ];
    memcpy( e.data, orig.data, orig.rows * orig.step );
}

HstPlane2DbClone::~HstPlane2DbClone( )
{
    delete [] e.data;
}


/*************************************************************
 * HstPlane2DbNull
 *************************************************************/
HstPlane2DbNull::HstPlane2DbNull( const int width, const int height )
{
    e.step = width;
    e.cols = width;
    e.rows = height;
    e.data = new uint8_t[ e.rows * e.step ];
    memset( e.data, 0, e.rows * e.step );
}

HstPlane2DbNull::~HstPlane2DbNull( )
{
    delete [] e.data;
}

} // namespace cctag


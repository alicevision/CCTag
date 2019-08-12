/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <sstream>

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
 * memset functions
 *************************************************************/
void memset_hst( void* ptr, uint8_t value, size_t bytes )
{
    memset( ptr, value, bytes );
}

void memset_dev( void* ptr, uint8_t value, size_t bytes )
{
    POP_CUDA_MEMSET_SYNC( ptr, value, bytes );
}

void memset_hst( void* ptr, uint8_t value, size_t bytes, cudaStream_t )
{
    memset( ptr, value, bytes );
}

void memset_dev( void* ptr, uint8_t value, size_t bytes, cudaStream_t stream )
{
    POP_CUDA_MEMSET_ASYNC( ptr, value, bytes, stream );
}

/*************************************************************
 * memcpy functions
 *************************************************************/
bool copy_hst_from_hst( void* dst, size_t , void* src, size_t ,
                        size_t w_bytes, size_t h )
{
    memcpy( dst, src, w_bytes*h );
    return true;
}

bool copy_hst_from_hst( void* dst, size_t , void* src, size_t ,
                        size_t w_bytes, size_t h, cudaStream_t )
{
    memcpy( dst, src, w_bytes*h );
    return true;
}

static inline
bool copy_x_from_x( void* dst, size_t dst_pitch,
                    void* src, size_t src_pitch,
                    size_t w_bytes, size_t h,
                    const cudaMemcpyKind kind,
                    const char* func )
{
    cudaError_t err;
    err = cudaMemcpy2D( dst, dst_pitch, src, src_pitch,
                        w_bytes, h,
                        kind );

    if( err == cudaSuccess ) return true;

    std::ostringstream ostr;
    ostr << "Memcpy failed in " << func << ", reason: " << cudaGetErrorString(err);
    POP_CUDA_FATAL_TEST( err, ostr.str() );
    return false;
}

bool copy_hst_from_dev( void* dst, size_t dst_pitch, void* src, size_t src_pitch,
                        size_t w_bytes, size_t h )
{
    return copy_x_from_x( dst, dst_pitch, src, src_pitch, w_bytes, h,
                          cudaMemcpyDeviceToHost, __FUNCTION__ );
}

bool copy_dev_from_hst( void* dst, size_t dst_pitch, void* src, size_t src_pitch,
                        size_t w_bytes, size_t h )
{
    return copy_x_from_x( dst, dst_pitch, src, src_pitch, w_bytes, h,
                          cudaMemcpyHostToDevice, __FUNCTION__ );
}

bool copy_dev_from_dev( void* dst, size_t dst_pitch, void* src, size_t src_pitch,
                        size_t w_bytes, size_t h )
{
    return copy_x_from_x( dst, dst_pitch, src, src_pitch, w_bytes, h,
                          cudaMemcpyDeviceToDevice, __FUNCTION__ );
}

static inline
bool copy_x_from_x( void* dst, size_t dst_pitch,
                    void* src, size_t src_pitch,
                    size_t w_bytes, size_t h,
                    cudaStream_t stream,
                    const cudaMemcpyKind kind,
                    const char* func )
{
    cudaError_t err;
    err = cudaMemcpy2DAsync( dst, dst_pitch, src, src_pitch,
                             w_bytes, h,
                             kind,
                             stream );

    if( err == cudaSuccess ) return true;

    std::ostringstream ostr;
    ostr << "Memcpy failed in " << func << ", reason: " << cudaGetErrorString(err);
    POP_CUDA_FATAL_TEST( err, ostr.str() );
    return false;
}

bool copy_hst_from_dev( void* dst, size_t dst_pitch, void* src, size_t src_pitch,
                        size_t w_bytes, size_t h, cudaStream_t stream )
{
    return copy_x_from_x( dst, dst_pitch, src, src_pitch, w_bytes, h, stream,
                          cudaMemcpyDeviceToHost, __FUNCTION__ );
}

bool copy_dev_from_hst( void* dst, size_t dst_pitch, void* src, size_t src_pitch,
                        size_t w_bytes, size_t h, cudaStream_t stream )
{
    return copy_x_from_x( dst, dst_pitch, src, src_pitch, w_bytes, h, stream,
                          cudaMemcpyHostToDevice, __FUNCTION__ );
}

bool copy_dev_from_dev( void* dst, size_t dst_pitch, void* src, size_t src_pitch,
                        size_t w_bytes, size_t h, cudaStream_t stream )
{
    return copy_x_from_x( dst, dst_pitch, src, src_pitch, w_bytes, h, stream,
                          cudaMemcpyDeviceToDevice, __FUNCTION__ );
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


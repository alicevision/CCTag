/*
 * Copyright 2016, Simula Research Laboratory
 *           2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <type_traits>
#include <cstdint>

// #include "cctag/cuda/cctag_cuda_runtime.h"

namespace cctag {

void* allocate_hst( size_t rows, size_t cols, size_t& step );
void* allocate_dev( size_t rows, size_t cols, size_t& step );
void release_hst( void* ptr );
void release_dev( void* ptr );

struct Hst { };
struct Dev { };

template<typename T, typename Ctx = Hst>
class Plane2D
{
public:
    __host__ __device__
    Plane2D( )
    { }

    template<typename S>
    __host__ __device__
    Plane2D( const Plane2D<S, Ctx>& orig )
    {
        data = (T*)orig.data;
        step = orig.step;
        cols = std::is_same<T,S>::value
             ? orig.cols
             : orig.step / sizeof(T);
        rows = orig.rows;
    }

    template<typename S>
    __host__ __device__
    Plane2D( size_t height, size_t width, const Plane2D<S, Ctx>& orig, size_t pitch )
        : data( (T*)orig.data )
        , step(pitch)
        , cols(width)
        , rows(height)
    { }

    __host__ __device__
    Plane2D( size_t height, size_t width, T* buf, size_t pitch )
        : data(buf)
        , step(pitch)
        , cols(width)
        , rows(height)
    { }

    __host__ __device__
    ~Plane2D( )
    {
        data = 0;
        step = 0;
        cols = 0;
        rows = 0;
    }

    __host__ __device__
    Plane2D& operator=( const Plane2D& orig )
    {
        data = orig.data;
        step = orig.step;
        cols = orig.cols;
        rows = orig.rows;
        return *this;
    }

    __host__ __device__
    T* ptr( size_t row )
    {
        return (T*)(((uint8_t*)data) + row * step);
    }

    __host__ __device__
    const T* ptr( size_t row ) const
    {
        return (T*)(((uint8_t*)data) + row * step);
    }

    __host__ __device__
    size_t elemSize() const
    {
        return sizeof(T);
    }

    __host__
    void allocate( size_t height, size_t width )
    {
        rows = height;
        cols = width;
        if( std::is_same<Ctx,Hst>::value )
            data = (T*)allocate_hst( rows, cols*sizeof(T), step );
        else
            data = (T*)allocate_dev( rows, cols*sizeof(T), step );
    }

    __host__
    void release( )
    {
        if( std::is_same<Ctx,Hst>::value )
            release_hst( data );
        else
            release_dev( data );
        data = 0;
        rows = cols = step = 0;
    }

    T*     data;
    size_t step;
    size_t cols;
    size_t rows;
};

using HstPlane2Db    = Plane2D<uint8_t,  Hst>;
using HstPlane2D16s  = Plane2D<int16_t,  Hst>;
using HstPlane2D32u  = Plane2D<uint32_t, Hst>;
using HstPlane2D32s  = Plane2D<int32_t,  Hst>;
using HstPlane2Db4   = Plane2D<uchar4,   Hst>;
using HstPlane2DInt2 = Plane2D<int2,     Hst>;
using HstPlane2Df    = Plane2D<float,    Hst>;

using DevPlane2Db    = Plane2D<uint8_t,  Dev>;
using DevPlane2D16s  = Plane2D<int16_t,  Dev>;
using DevPlane2D32u  = Plane2D<uint32_t, Dev>;
using DevPlane2D32s  = Plane2D<int32_t,  Dev>;
using DevPlane2Db4   = Plane2D<uchar4,   Dev>;
using DevPlane2DInt2 = Plane2D<int2,     Dev>;
using DevPlane2Df    = Plane2D<float,    Dev>;

struct HstPlane2DbClone
{
    HstPlane2Db e;

    __host__
    HstPlane2DbClone( const HstPlane2Db& orig );

    __host__
    ~HstPlane2DbClone( );

private:
    HstPlane2DbClone( );
    HstPlane2DbClone( const HstPlane2DbClone& );
    HstPlane2DbClone& operator=( const HstPlane2DbClone& );
};

struct HstPlane2DbNull
{
    HstPlane2Db e;

    __host__
    HstPlane2DbNull( const int width, const int height );

    __host__
    ~HstPlane2DbNull( );

private:
    HstPlane2DbNull( );
    HstPlane2DbNull( const HstPlane2DbNull& );
    HstPlane2DbNull& operator=( const HstPlane2DbNull& );
};

} // namespace cctag


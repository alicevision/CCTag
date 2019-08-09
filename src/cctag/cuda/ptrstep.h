/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <type_traits>
#include <cstdint>

#include "cctag/cuda/cctag_cuda_runtime.h"

namespace cctag {

template<typename T>
class PtrStepSz
{
public:
    __host__ __device__
    PtrStepSz( )
    { }

    template<typename S>
    __host__ __device__
    PtrStepSz( const PtrStepSz<S>& orig )
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
    PtrStepSz( size_t height, size_t width, const PtrStepSz<S>& orig, size_t pitch )
        : data( (T*)orig.data )
        , step(pitch)
        , cols(width)
        , rows(height)
    { }

    __host__ __device__
    PtrStepSz( size_t height, size_t width, T* buf, size_t pitch )
        : data(buf)
        , step(pitch)
        , cols(width)
        , rows(height)
    { }

    __host__ __device__
    ~PtrStepSz( )
    {
        data = 0;
        step = 0;
        cols = 0;
        rows = 0;
    }

    __host__ __device__
    PtrStepSz& operator=( const PtrStepSz& orig )
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

    T*     data;
    size_t step;
    size_t cols;
    size_t rows;
};

using PtrStepSzb         = PtrStepSz<uint8_t>;
using PtrStepSz16s       = PtrStepSz<int16_t>;
using PtrStepSz32u       = PtrStepSz<uint32_t>;
using PtrStepSz32s       = PtrStepSz<int32_t>;
using PtrStepSzb4        = PtrStepSz<uchar4>;
using PtrStepSzInt2      = PtrStepSz<int2>;
using PtrStepSzf         = PtrStepSz<float>;

struct PtrStepSzbClone
{
    PtrStepSzb e;

    __host__
    PtrStepSzbClone( const PtrStepSzb& orig );

    __host__
    ~PtrStepSzbClone( );

private:
    PtrStepSzbClone( );
    PtrStepSzbClone( const PtrStepSzbClone& );
    PtrStepSzbClone& operator=( const PtrStepSzbClone& );
};

struct PtrStepSzbNull
{
    PtrStepSzb e;

    __host__
    PtrStepSzbNull( const int width, const int height );

    __host__
    ~PtrStepSzbNull( );

private:
    PtrStepSzbNull( );
    PtrStepSzbNull( const PtrStepSzbNull& );
    PtrStepSzbNull& operator=( const PtrStepSzbNull& );
};

} // namespace cctag


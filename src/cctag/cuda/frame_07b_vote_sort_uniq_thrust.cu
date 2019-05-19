/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cuda.h>

#ifndef CUDA_VERSION
#error need a CUDA_VERSION string
#endif

#if CUDA_VERSION >= 8000

#include <cctag/cuda/cctag_cuda_runtime.h>
#include <limits>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>

#include "debug_macros.hpp"
#include "frame.h"
#include "framemeta.h"
#include "assist.h"

using namespace std;

namespace cctag
{

__host__
bool Frame::applyVoteSortUniq( )
{
    _inner_points.copySizeFromDevice( _stream, EdgeListWait );

    POP_CUDA_SYNC( _stream );

    if( _inner_points.host.size <= 0 ) {
        return false;
    }

    int sz = _inner_points.host.size;

    thrust::device_ptr<int> input_begin = thrust::device_pointer_cast( _inner_points.dev.ptr );
    thrust::device_ptr<int> input_end   = input_begin + sz;
    thrust::device_ptr<int> output_begin = thrust::device_pointer_cast( _interm_inner_points.dev.ptr );
    thrust::device_ptr<int> output_end;

    thrust::sort( input_begin, input_end );

    output_end = thrust::unique_copy( input_begin, input_end, output_begin );

    sz = output_end - output_begin;

    _meta.toDevice( List_size_interm_inner_points, sz, _stream );

    POP_CHK_CALL_IFSYNC;

    return true;
}

} // namespace cctag
#endif // CUDA_VERSION >= 8000


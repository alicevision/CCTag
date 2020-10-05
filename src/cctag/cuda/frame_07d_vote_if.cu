/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cuda.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#ifdef CCTAG_NO_THRUST_COPY_IF
#include <thrust/host_vector.h>
#endif

#include <iostream>
#include <algorithm>
#include <limits>
#include <cctag/cuda/cctag_cuda_runtime.h>
#include <stdio.h>
#include "debug_macros.hpp"
#include "debug_is_on_edge.h"

#include "frame.h"
#include "frameparam.h"
#include "assist.h"
#include "onoff.h"

using namespace std;

namespace cctag {

struct NumVotersIsGreaterEqual
{
    DevEdgeList<TriplePoint> _array;

    __host__ __device__
    __forceinline__
    NumVotersIsGreaterEqual( DevEdgeList<TriplePoint> _d_array )
        : _array( _d_array )
    {}

    __device__
    __forceinline__
    bool operator()(const int &a) const {
        return (_array.ptr[a]._winnerSize >= tagParam.minVotesToSelectCandidate );
    }
};

__host__
bool Frame::applyVoteIf( )
{
    if( _interm_inner_points.host.size == 0 ) {
        return false;
    }

    int                     sz           = _interm_inner_points.host.size;
    thrust::device_ptr<int> input_begin  = thrust::device_pointer_cast( _interm_inner_points.dev.ptr );
    thrust::device_ptr<int> input_end    = input_begin + sz;
    thrust::device_ptr<int> output_begin = thrust::device_pointer_cast( _inner_points.dev.ptr );
    thrust::device_ptr<int> output_end;

    NumVotersIsGreaterEqual select_op( _voters.dev );

#ifdef CCTAG_NO_THRUST_COPY_IF
    //
    // There are reports that the Thrust::copy_if fails when you generated code with CUDA 7.0 and run it only
    // a 2nd gen Maxwell card (e.g. GTX 980 and GTX 980 Ti). Also, the GTX 1080 seems to be quite similar to
    // the GTX 980 and may be affected as well.
    // The following code moves everything to host before copy_if, which circumvents the problem but is much
    // slower. Make sure that the condition for activating it is very strict.
    //
    thrust::host_vector<int> input_host(sz);
    thrust::host_vector<int> output_host(sz);
    thrust::host_vector<int>::iterator output_host_end;

    thrust::copy( input_begin, input_end, input_host.begin() );
    output_host_end = thrust::copy_if( input_host.begin(), input_host.end(), output_host.begin(), select_op );
    thrust::copy( output_host.begin(), output_host_end, output_begin );
    sz = output_host_end - output_host.begin();
    output_end = output_begin + sz;
#else
    output_end = thrust::copy_if( input_begin, input_end, output_begin, select_op );

    sz = output_end - output_begin;
#endif

    POP_CUDA_SYNC( _stream );
    _meta.toDevice( List_size_inner_points, sz, _stream );
    _inner_points.host.size = sz;

    return true;
}

} // namespace cctag


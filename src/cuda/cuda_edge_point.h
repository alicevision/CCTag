/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

namespace popart
{

class CudaEdgePoint : public Vector3s
{
    short3   _coord;
    float2   _grad;
    float    _normGrad;
    float    _flowLength;
    uint64_t _processed;
    int      _isMax;
    int      _nSegmentOut;

    __device__ inline
    void init( short x, short y, float dx, float dy )
    {
        _coord       = make_short3( x, y, 1 ); // done after step 05
        _grad        = make_float2( dx, dy );  // done after step 05
        _normGrad    = __hypotf( dx, dy );     // done after step 05
        _flowLength  = 0;
        _processed   = 0;
        _isMax       = -1;
        _nSegmentOut = -1;
    }
};

} // popart


#pragma once

#include <cuda_runtime.h>
#include <string>
#include <sstream>
#include <iostream>

namespace popart {

struct CudaEdgePoint
{
    short2  _coord;
    float2  _grad;
    float   _normGrad;
    float   _flowLength;

    int     _dev_winnerSize;
    int     _dev_befor;      // index into _edgepoints
    int     _dev_after;      // index into _edgepoints

    __device__ inline void init( );
    __device__ inline void init( short x, short y );
    __device__ inline void init( short x, short y, short dx, short dy );

    __host__ void debug_out( std::ostream& ostr ) const;
};

__device__
inline void CudaEdgePoint::init( )
{
    _coord          = make_short2( 0, 0 );
    _grad           = make_float2( 0.0f, 0.0f );
    _normGrad       = 0.0f;
    _flowLength     = 0.0f;
    _dev_winnerSize = 0;
    _dev_befor      = 0;
    _dev_after      = 0;
}

__device__
inline void CudaEdgePoint::init( short x, short y )
{
    _coord          = make_short2( x, y );
    _grad           = make_float2( 0.0f, 0.0f );
    _normGrad       = 0.0f;
    _flowLength     = 0.0f;
    _dev_winnerSize = 0;
    _dev_befor      = 0;
    _dev_after      = 0;
}

__device__
inline void CudaEdgePoint::init( short x, short y, short dx, short dy )
{
    _coord          = make_short2( x, y );
    _grad           = make_float2( dx, dy );
    _normGrad       = 0.0f;
    _flowLength     = 0.0f;
    _dev_winnerSize = 0;
    _dev_befor      = 0;
    _dev_after      = 0;
}

}; // namespace popart


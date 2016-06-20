#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "cuda/edge_list.h"

__host__
inline bool tpcmp( const int2& l, const int2& r )
{
    return ( l.y < r.y || ( l.y == r.y && l.x < r.x ) );
}

__host__
inline bool tpcmp( const short2& l, const short2& r )
{
    return ( l.y < r.y || ( l.y == r.y && l.x < r.x ) );
}

class int2cmp
{
public:
    __host__
    inline bool operator()( const int2& l, const int2& r )
    {
        return tpcmp( l, r );
    }
    __host__
    inline bool operator()( const short2& l, const short2& r )
    {
        return tpcmp( l, r );
    }
};

std::ostream& operator<<( std::ostream& ostr, const int2& v );
std::ostream& operator<<( std::ostream& ostr, const short2& v );


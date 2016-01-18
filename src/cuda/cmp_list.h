#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include "cuda/triple_point.h"
#include "cuda/edge_list.h"

__host__
inline bool tpcmp( const int2& l, const int2& r )
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
};

class tp_cmp
{
public:
    __host__
    inline bool operator()( const popart::TriplePoint& l, const popart::TriplePoint& r )
    {
        return tpcmp( l.coord, r.coord );
    }
};

class vote_index_sort
{
public:
    __host__
    vote_index_sort( const popart::HostEdgeList<popart::TriplePoint>& voters );

    __host__
    inline bool operator()( const int l, const int r )
    {
        bool val = false;
        if( _voters.ptr[l]._winnerSize > _voters.ptr[r]._winnerSize ) {
            val = true;
        } else if( _voters.ptr[l]._winnerSize == _voters.ptr[r]._winnerSize ) {
            val = tpcmp( _voters.ptr[l].coord, _voters.ptr[r].coord );
        }
        return val;
    }

private:
    const popart::HostEdgeList<popart::TriplePoint>& _voters;
};

std::ostream& operator<<( std::ostream& ostr, const int2& v );


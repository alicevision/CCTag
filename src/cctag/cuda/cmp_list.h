/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/cuda/cctag_cuda_runtime.h>
#include <iostream>
#include "cctag/cuda/triple_point.h"
#include "cctag/cuda/edge_list.h"

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
    inline bool operator()( const cctag::TriplePoint& l, const cctag::TriplePoint& r )
    {
        return tpcmp( l.coord, r.coord );
    }
};

class vote_index_sort
{
public:
    __host__
    vote_index_sort( const cctag::HostEdgeList<cctag::TriplePoint>& voters );

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
    const cctag::HostEdgeList<cctag::TriplePoint>& _voters;
};

std::ostream& operator<<( std::ostream& ostr, const int2& v );


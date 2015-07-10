#pragma once

#include <cuda_runtime.h>
#include <string>
#include <sstream>

namespace popart {

/*************************************************************
 * TriplePoint
 * A simplified version of EdgePoint in the C++ code.
 *************************************************************/
struct TriplePoint
{
    int2  coord;
    int2  befor;
    int2  after;

    // in the original code, chosen keeps list of voters
    // no possible here; we must invert this
    int   my_vote;
    float chosen_flow_length;

    int   _winnerSize;
    float _flowLength;
#ifndef NDEBUG
    int   _coords_idx;
    int2  _coords[10];

    __device__
    inline void debug_init( ) {
        _coords_idx = 0;
    }
    __device__
    inline void debug_add( int2 c ) {
        if( _coords_idx >= 10 ) return;
        _coords[_coords_idx].x = c.x;
        _coords[_coords_idx].y = c.y;
        _coords_idx++;
    }

    __host__
    inline std::string debug_out( ) const {
        std::ostringstream ostr;
        ostr << coord.x << " " << coord.y << " "
             << befor.x << " " << befor.y << " "
             << after.x << " " << after.y;
        if( _coords_idx != 0 ) {
            for( int i=0; i<_coords_idx; i++ ) {
                ostr << " (" << _coords[i].x << "," << _coords[i].y << ")";
            }
        }
        return ostr.str();

    }
#endif
};

}; // namespace popart

